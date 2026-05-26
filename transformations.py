"""
transformations.py – Convert raw weather into power-market trading signals.

Modules
-------
wind_power_proxy       – piece-wise linear power curve → capacity factor 0-1
solar_proxy            – radiation + cloud cover → CF 0-1
demand_signal          – HDD/CDD → normalised demand pressure –1..+1
node_score             – weighted composite score per node
country_scores         – aggregate to country level + subregion breakdown
ramp_detection         – flag sharp changes in the 0-72h window
spread_signals         – inter-country differential signals
trade_insights         – auto-generate natural-language insight bullets
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

from config import (
    NODES,
    WIND_CUT_IN, WIND_RATED, WIND_CUT_OUT,
    BALANCE_TEMP, COLD_THRESHOLD, HEAT_THRESHOLD,
    WEIGHTS,
    SCORE_BULL_THRESHOLD, SCORE_BEAR_THRESHOLD,
    WIND_RAMP_THRESHOLD_MW, SOLAR_RAMP_THRESHOLD_MW, TEMP_RAMP_THRESHOLD,
    SPREAD_PAIRS,
    BULL_COLOR, BEAR_COLOR, NEUTRAL_COLOR,
)


# ══════════════════════════════════════════════════════════════════════════════
# 1. WIND POWER PROXY
# ══════════════════════════════════════════════════════════════════════════════

def wind_power_proxy(windspeed_ms: pd.Series) -> pd.Series:
    """
    Piece-wise linear wind power curve → capacity factor [0, 1].

    Below cut-in  (< 3 m/s):  CF = 0
    Ramp region   (3–12 m/s): CF = linear 0→1
    Rated region  (12–25 m/s): CF = 1
    Above cut-out (> 25 m/s): CF = 0  (storm shutdown)

    Uses 100m hub-height wind speed where available, falls back to 10m.
    """
    ws = windspeed_ms.clip(lower=0)
    cf = np.where(
        ws < WIND_CUT_IN, 0.0,
        np.where(
            ws <= WIND_RATED,
            (ws - WIND_CUT_IN) / (WIND_RATED - WIND_CUT_IN),
            np.where(ws <= WIND_CUT_OUT, 1.0, 0.0),
        ),
    )
    return pd.Series(cf, index=windspeed_ms.index, name="wind_cf")


# ══════════════════════════════════════════════════════════════════════════════
# 2. SOLAR GENERATION PROXY
# ══════════════════════════════════════════════════════════════════════════════

_SOLAR_PEAK_W_M2 = 900.0  # reference clear-sky irradiance

def solar_proxy(radiation: pd.Series, cloudcover: pd.Series) -> pd.Series:
    """
    Solar capacity factor proxy [0, 1].

    CF = (radiation / peak) * (1 – cloudcover_fraction * 0.75)

    The 0.75 factor reflects that diffuse radiation still reaches panels
    through partial cloud cover.
    """
    rad_norm  = (radiation.clip(lower=0) / _SOLAR_PEAK_W_M2).clip(upper=1.0)
    cloud_pen = 1.0 - (cloudcover.clip(0, 100) / 100.0) * 0.75
    cf = (rad_norm * cloud_pen).clip(0.0, 1.0)
    return pd.Series(cf, index=radiation.index, name="solar_cf")


# ══════════════════════════════════════════════════════════════════════════════
# 3. TEMPERATURE-BASED DEMAND SIGNAL
# ══════════════════════════════════════════════════════════════════════════════

def demand_signal(temperature: pd.Series) -> pd.Series:
    """
    Normalised demand pressure signal in [-1, +1].

    Positive (bullish price): cold-driven heating demand  → HDD > 0
    Negative (bearish price): heat-driven cooling demand  → CDD > 0
    Zero: near balance temperature

    HDD = max(0, BALANCE_TEMP – T)
    CDD = max(0, T – BALANCE_TEMP)

    We normalise each by a plausible peak value (10°C deviation) then
    net them: demand_signal = (HDD – CDD) / 10, clipped to [-1, 1].
    """
    hdd = (BALANCE_TEMP - temperature).clip(lower=0)
    cdd = (temperature - BALANCE_TEMP).clip(lower=0)
    sig = ((hdd - cdd) / 10.0).clip(-1.0, 1.0)
    return pd.Series(sig, index=temperature.index, name="demand_signal")


# ══════════════════════════════════════════════════════════════════════════════
# 4. PER-NODE ENRICHMENT
# ══════════════════════════════════════════════════════════════════════════════

def enrich_node(df: pd.DataFrame) -> pd.DataFrame:
    """Add proxy columns to a single-node DataFrame."""
    # Use 100m wind if available and sensible, else fall back to 10m
    ws = df.get("windspeed_100m", df["windspeed_10m"]).fillna(df["windspeed_10m"])
    df = df.copy()
    df["wind_cf"]       = wind_power_proxy(ws)
    df["solar_cf"]      = solar_proxy(df["shortwave_radiation"], df["cloudcover"])
    df["demand_sig"]    = demand_signal(df["temperature_2m"])
    df["hdd"]           = (BALANCE_TEMP - df["temperature_2m"]).clip(lower=0)
    df["cdd"]           = (df["temperature_2m"] - BALANCE_TEMP).clip(lower=0)

    # Composite score:  wind↑ & solar↑  → supply surplus → bearish price
    #                   demand↑          → demand pressure → bullish price
    df["score"] = (
        -WEIGHTS["wind"]   * df["wind_cf"]    # supply surplus is bearish
        - WEIGHTS["solar"] * df["solar_cf"]   # supply surplus is bearish
        + WEIGHTS["demand"]* df["demand_sig"] # demand pressure is bullish
    )
    return df


def enrich_all(node_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    return {nid: enrich_node(df) for nid, df in node_data.items()}


# ══════════════════════════════════════════════════════════════════════════════
# 5. COUNTRY & SUBREGION AGGREGATION
# ══════════════════════════════════════════════════════════════════════════════

def _score_label(score: float) -> Tuple[str, str]:
    if score >= SCORE_BULL_THRESHOLD:
        return "🟢 BULLISH", BULL_COLOR
    elif score <= SCORE_BEAR_THRESHOLD:
        return "🔴 BEARISH", BEAR_COLOR
    else:
        return "🟡 NEUTRAL", NEUTRAL_COLOR


def country_summary(enriched: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Return a DataFrame with one row per country/subregion showing:
    - mean score over next 24h
    - mean score over full 72h
    - current hour values
    - signal label
    """
    rows = []
    for node_id, df in enriched.items():
        if df.empty:
            continue
        meta   = NODES[node_id]
        now_val = df.iloc[0] if len(df) else None
        h24    = df.head(24)
        h72    = df

        row = {
            "node_id":     node_id,
            "label":       meta["label"],
            "country":     meta["country"],
            "subregion":   meta["subregion"],
            # Current snapshot
            "temp_now":    round(now_val["temperature_2m"], 1) if now_val is not None else np.nan,
            "wind_cf_now": round(now_val["wind_cf"], 2)        if now_val is not None else np.nan,
            "solar_cf_now":round(now_val["solar_cf"], 2)       if now_val is not None else np.nan,
            "wind_ms_now": round(now_val.get("windspeed_100m", now_val["windspeed_10m"]), 1) if now_val is not None else np.nan,
            # Averages
            "score_24h":  round(h24["score"].mean(), 3),
            "score_72h":  round(h72["score"].mean(), 3),
            "wind_24h":   round(h24["wind_cf"].mean(), 2),
            "solar_24h":  round(h24["solar_cf"].mean(), 2),
            "demand_24h": round(h24["demand_sig"].mean(), 2),
            "hdd_24h":    round(h24["hdd"].mean(), 1),
            "cdd_24h":    round(h24["cdd"].mean(), 1),
        }
        label, color = _score_label(row["score_24h"])
        row["signal"]       = label
        row["signal_color"] = color
        rows.append(row)

    return pd.DataFrame(rows)


def country_level(summary: pd.DataFrame) -> pd.DataFrame:
    """Aggregate node-level summary to country level."""
    agg = (
        summary.groupby("country")
        .agg(
            score_24h=("score_24h", "mean"),
            score_72h=("score_72h", "mean"),
            wind_24h=("wind_24h", "mean"),
            solar_24h=("solar_24h", "mean"),
            demand_24h=("demand_24h", "mean"),
            hdd_24h=("hdd_24h", "mean"),
            cdd_24h=("cdd_24h", "mean"),
            temp_now=("temp_now", "mean"),
            wind_ms_now=("wind_ms_now", "mean"),
        )
        .reset_index()
    )
    labels, colors = zip(*agg["score_24h"].apply(_score_label))
    agg["signal"]       = labels
    agg["signal_color"] = colors
    return agg.sort_values("score_24h")


# ══════════════════════════════════════════════════════════════════════════════
# 6. RAMP DETECTION
# ══════════════════════════════════════════════════════════════════════════════

def detect_ramps(enriched: Dict[str, pd.DataFrame]) -> List[dict]:
    """
    Scan next 72h for sharp changes in wind CF, solar CF, temperature.
    Returns list of ramp event dicts.
    """
    events = []
    for node_id, df in enriched.items():
        if len(df) < 4:
            continue
        meta = NODES[node_id]

        # 3-hour rolling diff
        w_diff  = df["wind_cf"].diff(3).abs()
        s_diff  = df["solar_cf"].diff(3).abs()
        t_diff  = df["temperature_2m"].diff(3).abs()

        for ts, row in df.iterrows():
            idx = df.index.get_loc(ts)
            if idx < 3:
                continue
            wd = w_diff.iloc[idx]
            sd = s_diff.iloc[idx]
            td = t_diff.iloc[idx]

            if wd >= WIND_RAMP_THRESHOLD_MW:
                direction = "▲ ramp UP" if df["wind_cf"].diff(3).iloc[idx] > 0 else "▼ ramp DOWN"
                events.append({
                    "time": ts, "node": meta["label"], "country": meta["country"],
                    "type": "Wind", "direction": direction,
                    "magnitude": round(wd, 2), "severity": "HIGH" if wd > 0.4 else "MOD",
                })
            if sd >= SOLAR_RAMP_THRESHOLD_MW:
                direction = "▲ ramp UP" if df["solar_cf"].diff(3).iloc[idx] > 0 else "▼ ramp DOWN"
                events.append({
                    "time": ts, "node": meta["label"], "country": meta["country"],
                    "type": "Solar", "direction": direction,
                    "magnitude": round(sd, 2), "severity": "HIGH" if sd > 0.5 else "MOD",
                })
            if td >= TEMP_RAMP_THRESHOLD:
                direction = "▲ rising" if df["temperature_2m"].diff(3).iloc[idx] > 0 else "▼ falling"
                events.append({
                    "time": ts, "node": meta["label"], "country": meta["country"],
                    "type": "Temp", "direction": direction,
                    "magnitude": round(td, 1), "severity": "MOD",
                })

    # Deduplicate and sort by time
    events.sort(key=lambda e: e["time"])
    return events


# ══════════════════════════════════════════════════════════════════════════════
# 7. SPREAD SIGNALS
# ══════════════════════════════════════════════════════════════════════════════

def spread_signals(country_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute pairwise score differentials for defined spread pairs.
    Positive spread → A more bullish than B (higher prices expected in A).
    """
    rows = []
    scores = country_df.set_index("country")["score_24h"].to_dict()
    for cA, cB, name in SPREAD_PAIRS:
        sA = scores.get(cA)
        sB = scores.get(cB)
        if sA is None or sB is None:
            continue
        diff = round(sA - sB, 3)
        label, color = _score_label(diff * 2)  # amplify for readability
        rows.append({
            "pair": name,
            "country_A": cA,
            "country_B": cB,
            "score_A": sA,
            "score_B": sB,
            "spread": diff,
            "signal": label,
            "signal_color": color,
        })
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# 8. INTRA-COUNTRY SPREAD (e.g., North vs South Germany)
# ══════════════════════════════════════════════════════════════════════════════

def intra_country_spreads(summary: pd.DataFrame) -> pd.DataFrame:
    """North vs South (or sub-region) score differential per country."""
    rows = []
    for country, grp in summary.groupby("country"):
        sub_scores = grp.groupby("subregion")["score_24h"].mean()
        if len(sub_scores) < 2:
            continue
        subs = sub_scores.index.tolist()
        for i, a in enumerate(subs):
            for b in subs[i+1:]:
                diff = round(sub_scores[a] - sub_scores[b], 3)
                label, color = _score_label(diff * 2)
                rows.append({
                    "country": country,
                    "spread": f"{a} vs {b}",
                    "score_A": sub_scores[a],
                    "score_B": sub_scores[b],
                    "diff": diff,
                    "signal": label,
                    "signal_color": color,
                })
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# 9. TRADE INSIGHT GENERATOR
# ══════════════════════════════════════════════════════════════════════════════

def generate_insights(
    country_df: pd.DataFrame,
    spreads_df: pd.DataFrame,
    ramps: List[dict],
    node_summary: pd.DataFrame,
) -> List[dict]:
    """
    Auto-generate actionable trader-style insights.
    Returns list of dicts: {icon, text, color, priority}.
    """
    insights = []
    c = country_df.set_index("country")

    # ── Wind insights ──────────────────────────────────────────────────────
    low_wind = country_df[country_df["wind_24h"] < 0.15]
    for _, row in low_wind.iterrows():
        insights.append({
            "icon": "💨",
            "text": f"**{row['country']}**: Wind generation very low (CF {row['wind_24h']:.0%}) "
                    f"over next 24h → bullish supply squeeze. Watch baseload premium.",
            "color": BULL_COLOR,
            "priority": 1,
        })

    high_wind = country_df[country_df["wind_24h"] > 0.70]
    for _, row in high_wind.iterrows():
        insights.append({
            "icon": "🌬️",
            "text": f"**{row['country']}**: Exceptional wind output (CF {row['wind_24h']:.0%}) "
                    f"→ price suppression likely. Bearish intraday, esp. peak vs off-peak.",
            "color": BEAR_COLOR,
            "priority": 1,
        })

    # ── Solar insights ──────────────────────────────────────────────────────
    high_solar = country_df[(country_df["solar_24h"] > 0.40) & (country_df["country"].isin(["ES","IT","PT","FR"]))]
    for _, row in high_solar.iterrows():
        insights.append({
            "icon": "☀️",
            "text": f"**{row['country']}**: Strong solar generation expected (CF {row['solar_24h']:.0%}). "
                    f"Mid-day duck curve risk — bearish 10:00–15:00, bullish morning/evening ramp.",
            "color": BEAR_COLOR,
            "priority": 2,
        })

    # ── Cold demand insights ────────────────────────────────────────────────
    cold_nodes = node_summary[node_summary["temp_now"] < COLD_THRESHOLD]
    if not cold_nodes.empty:
        cold_countries = cold_nodes["country"].unique()
        avg_temps = cold_nodes.groupby("country")["temp_now"].mean().round(1)
        for ctry in cold_countries:
            t = avg_temps.get(ctry, "?")
            insights.append({
                "icon": "🥶",
                "text": f"**{ctry}**: Current temps {t}°C → elevated heating demand. "
                        f"HDD signal active. Expect residential load pickup overnight.",
                "color": BULL_COLOR,
                "priority": 1,
            })

    # ── Heat demand insights ────────────────────────────────────────────────
    hot_nodes = node_summary[node_summary["temp_now"] > HEAT_THRESHOLD]
    if not hot_nodes.empty:
        hot_countries = hot_nodes["country"].unique()
        avg_temps = hot_nodes.groupby("country")["temp_now"].mean().round(1)
        for ctry in hot_countries:
            t = avg_temps.get(ctry, "?")
            insights.append({
                "icon": "🌡️",
                "text": f"**{ctry}**: Heat at {t}°C driving A/C load. "
                        f"CDD signal active. Peak hours 13:00–18:00 at risk of demand spikes.",
                "color": BULL_COLOR,
                "priority": 2,
            })

    # ── Ramp insights ───────────────────────────────────────────────────────
    high_sev_ramps = [r for r in ramps if r["severity"] == "HIGH"][:5]
    for r in high_sev_ramps:
        ts_str = r["time"].strftime("%d-%b %H:%M UTC")
        insights.append({
            "icon": "⚡",
            "text": f"**{r['country']} – {r['node']}**: {r['type']} {r['direction']} "
                    f"(Δ{r['magnitude']:.0%} CF) at {ts_str}. "
                    f"Sharp generation transition — intraday volatility risk.",
            "color": NEUTRAL_COLOR,
            "priority": 1,
        })

    # ── Spread insights ─────────────────────────────────────────────────────
    if not spreads_df.empty:
        wide_spreads = spreads_df[spreads_df["spread"].abs() > 0.15]
        for _, row in wide_spreads.iterrows():
            direction = "A outperforming B" if row["spread"] > 0 else "B outperforming A"
            insights.append({
                "icon": "↔️",
                "text": f"**{row['pair']}**: Spread of {row['spread']:+.2f} — {direction}. "
                        f"Cross-border arbitrage window may be open.",
                "color": NEUTRAL_COLOR,
                "priority": 3,
            })

    # ── Nordic hydro proxy ──────────────────────────────────────────────────
    nordic = country_df[country_df["country"] == "NO"]
    if not nordic.empty and nordic.iloc[0]["wind_24h"] > 0.50:
        insights.append({
            "icon": "💧",
            "text": "**Norway**: Strong wind reducing hydro dispatch pressure. "
                    "Reservoir spill risk low. Potential for increased NO→DE exports.",
            "color": BEAR_COLOR,
            "priority": 3,
        })

    # Sort by priority (1 = most important)
    insights.sort(key=lambda x: x["priority"])
    return insights[:12]  # cap at 12 bullets


# ══════════════════════════════════════════════════════════════════════════════
# 10. TIMESERIES HELPERS FOR CHARTING
# ══════════════════════════════════════════════════════════════════════════════

def country_timeseries(enriched: Dict[str, pd.DataFrame], country_code: str) -> pd.DataFrame:
    """Average hourly timeseries for a given country code."""
    frames = [df for nid, df in enriched.items() if NODES[nid]["country"] == country_code]
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames)
    return combined.groupby(combined.index)[["wind_cf", "solar_cf", "demand_sig", "temperature_2m", "score"]].mean()


def all_countries_score_ts(enriched: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return wide-form score timeseries: index=time, columns=country codes."""
    country_codes = list({NODES[nid]["country"] for nid in enriched})
    series = {}
    for cc in country_codes:
        ts = country_timeseries(enriched, cc)
        if not ts.empty:
            series[cc] = ts["score"]
    if not series:
        return pd.DataFrame()
    return pd.DataFrame(series).sort_index()

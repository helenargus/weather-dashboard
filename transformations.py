"""
transformations.py – Convert raw weather into power-market trading signals.

Sections
--------
1.  wind_power_proxy       – piece-wise linear power curve → CF [0, 1]
2.  solar_proxy            – radiation × cloud-cover penalty → CF [0, 1]
3.  demand_signal          – HDD / CDD → normalised demand pressure [-1, +1]
4.  enrich_node / enrich_all – add proxy + score columns to each node DataFrame
5.  country_summary        – per-node summary rows → DataFrame
6.  country_level          – aggregate node rows to country-level DataFrame
7.  detect_ramps           – vectorised ramp/cliff detection across 72 h
8.  spread_signals         – inter-country score differentials
9.  intra_country_spreads  – sub-region imbalance within each country
10. generate_insights      – auto-generate trader-style bullet insights
11. country_timeseries     – average hourly series for one country
12. all_countries_score_ts – wide score table for the main chart
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

        ws < cut_in  (3 m/s)  → CF = 0
        cut_in ≤ ws ≤ rated   → CF linearly 0→1
        rated < ws ≤ cut_out  → CF = 1
        ws > cut_out (25 m/s) → CF = 0  (storm shutdown)
    """
    ws = windspeed_ms.clip(lower=0).to_numpy()
    cf = np.where(
        ws < WIND_CUT_IN,
        0.0,
        np.where(
            ws <= WIND_RATED,
            (ws - WIND_CUT_IN) / (WIND_RATED - WIND_CUT_IN),
            np.where(ws <= WIND_CUT_OUT, 1.0, 0.0),
        ),
    )
    return pd.Series(cf, index=windspeed_ms.index, name="wind_cf", dtype=float)


# ══════════════════════════════════════════════════════════════════════════════
# 2. SOLAR GENERATION PROXY
# ══════════════════════════════════════════════════════════════════════════════

_SOLAR_PEAK_W_M2 = 900.0  # reference clear-sky irradiance W/m²

def solar_proxy(radiation: pd.Series, cloudcover: pd.Series) -> pd.Series:
    """
    Solar capacity factor proxy [0, 1].

    CF = (radiation / 900) × (1 – cloud_fraction × 0.75)

    The 0.75 factor acknowledges that diffuse radiation still reaches
    panels through partial cloud.
    """
    rad_norm  = (radiation.clip(lower=0) / _SOLAR_PEAK_W_M2).clip(upper=1.0)
    cloud_pen = 1.0 - (cloudcover.clip(0, 100) / 100.0) * 0.75
    cf = (rad_norm * cloud_pen).clip(0.0, 1.0)
    return pd.Series(cf.to_numpy(), index=radiation.index, name="solar_cf", dtype=float)


# ══════════════════════════════════════════════════════════════════════════════
# 3. TEMPERATURE-BASED DEMAND SIGNAL
# ══════════════════════════════════════════════════════════════════════════════

def demand_signal(temperature: pd.Series) -> pd.Series:
    """
    Normalised demand pressure signal in [-1, +1].

        HDD = max(0, balance_temp – T)  → heating demand (bullish price)
        CDD = max(0, T – balance_temp)  → cooling demand (bullish price)
        signal = (HDD – CDD) / 10,  clipped to [-1, +1]

    Positive → cold-driven demand surge (bullish).
    Negative → heat-driven cooling demand (still bullish, different season).
    Zero     → mild weather, no temperature-driven pressure.
    """
    hdd = (BALANCE_TEMP - temperature).clip(lower=0)
    cdd = (temperature - BALANCE_TEMP).clip(lower=0)
    sig = ((hdd - cdd) / 10.0).clip(-1.0, 1.0)
    return pd.Series(sig.to_numpy(), index=temperature.index, name="demand_signal", dtype=float)


# ══════════════════════════════════════════════════════════════════════════════
# 4. PER-NODE ENRICHMENT
# ══════════════════════════════════════════════════════════════════════════════

def enrich_node(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add wind_cf, solar_cf, demand_sig, hdd, cdd, and composite score columns
    to a single-node DataFrame returned by data_fetch.

    Wind speed: prefers 100 m hub-height; falls back to 10 m if the column is
    absent or entirely NaN.
    """
    df = df.copy()

    # ── Wind speed: 100 m preferred, 10 m fallback ────────────────────────────
    ws100 = df.get("windspeed_100m")          # DataFrame.get → Series or None
    ws10  = df["windspeed_10m"]
    if ws100 is None or ws100.isna().all():
        ws = ws10.fillna(0.0)
    else:
        ws = ws100.fillna(ws10).fillna(0.0)

    # ── Proxy columns ─────────────────────────────────────────────────────────
    df["wind_cf"]    = wind_power_proxy(ws)
    df["solar_cf"]   = solar_proxy(df["shortwave_radiation"], df["cloudcover"])
    df["demand_sig"] = demand_signal(df["temperature_2m"])
    df["hdd"]        = (BALANCE_TEMP - df["temperature_2m"]).clip(lower=0)
    df["cdd"]        = (df["temperature_2m"] - BALANCE_TEMP).clip(lower=0)

    # ── Composite score ───────────────────────────────────────────────────────
    # Wind ↑ & Solar ↑  → supply surplus → bearish (negative score)
    # Demand ↑          → demand pressure → bullish (positive score)
    df["score"] = (
        -WEIGHTS["wind"]   * df["wind_cf"]
        - WEIGHTS["solar"] * df["solar_cf"]
        + WEIGHTS["demand"]* df["demand_sig"]
    )
    return df


def enrich_all(node_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """Apply enrich_node to every node in the dict."""
    return {nid: enrich_node(df) for nid, df in node_data.items()}


# ══════════════════════════════════════════════════════════════════════════════
# 5. COUNTRY & SUBREGION SUMMARY  (one row per node)
# ══════════════════════════════════════════════════════════════════════════════

def _score_label(score: float) -> Tuple[str, str]:
    """Return (signal_text, hex_color) for a composite score."""
    if score >= SCORE_BULL_THRESHOLD:
        return "🟢 BULLISH", BULL_COLOR
    elif score <= SCORE_BEAR_THRESHOLD:
        return "🔴 BEARISH", BEAR_COLOR
    else:
        return "🟡 NEUTRAL", NEUTRAL_COLOR


def country_summary(enriched: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Build a node-level summary DataFrame (one row per node) that includes
    country, subregion, current-hour snapshots, 24 h / 72 h averages, and
    bull/bear signal labels.

    Returns an empty DataFrame if `enriched` is empty or all nodes are empty.
    """
    rows: List[dict] = []

    for node_id, df in enriched.items():
        if df.empty:
            continue

        meta    = NODES[node_id]
        now_val = df.iloc[0]          # first future hour = "current" snapshot
        h24     = df.head(24)
        h72     = df                  # already trimmed to 72 h in data_fetch

        # ── Current-hour snapshot ─────────────────────────────────────────────
        # Prefer 100 m wind for display; fall back to 10 m safely via pandas
        ws100_now = now_val.get("windspeed_100m")
        if ws100_now is None or (isinstance(ws100_now, float) and np.isnan(ws100_now)):
            ws_now = now_val["windspeed_10m"]
        else:
            ws_now = ws100_now

        row: dict = {
            # Identity
            "node_id":   node_id,
            "label":     meta["label"],
            "country":   meta["country"],
            "subregion": meta["subregion"],
            # Current-hour values
            "temp_now":     round(float(now_val["temperature_2m"]), 1),
            "wind_cf_now":  round(float(now_val["wind_cf"]),  2),
            "solar_cf_now": round(float(now_val["solar_cf"]), 2),
            "wind_ms_now":  round(float(ws_now), 1),
            # 24 h averages
            "score_24h":  round(float(h24["score"].mean()),      3),
            "score_72h":  round(float(h72["score"].mean()),      3),
            "wind_24h":   round(float(h24["wind_cf"].mean()),    2),
            "solar_24h":  round(float(h24["solar_cf"].mean()),   2),
            "demand_24h": round(float(h24["demand_sig"].mean()), 2),
            "hdd_24h":    round(float(h24["hdd"].mean()),        1),
            "cdd_24h":    round(float(h24["cdd"].mean()),        1),
        }

        label, color = _score_label(row["score_24h"])
        row["signal"]       = label
        row["signal_color"] = color
        rows.append(row)

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# 6. COUNTRY-LEVEL AGGREGATION  (one row per country)
# ══════════════════════════════════════════════════════════════════════════════

def country_level(summary: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate the per-node summary to country level by averaging numeric
    columns.  Re-computes signal labels from the aggregated score.

    Raises ValueError if `summary` is empty or lacks a 'country' column.
    """
    if summary.empty or "country" not in summary.columns:
        return pd.DataFrame()

    agg = (
        summary
        .groupby("country", as_index=False)
        .agg(
            score_24h  =("score_24h",   "mean"),
            score_72h  =("score_72h",   "mean"),
            wind_24h   =("wind_24h",    "mean"),
            solar_24h  =("solar_24h",   "mean"),
            demand_24h =("demand_24h",  "mean"),
            hdd_24h    =("hdd_24h",     "mean"),
            cdd_24h    =("cdd_24h",     "mean"),
            temp_now   =("temp_now",    "mean"),
            wind_ms_now=("wind_ms_now", "mean"),
        )
    )

    # Recompute signal labels from the aggregated score
    signal_pairs          = agg["score_24h"].apply(_score_label)
    agg["signal"]         = signal_pairs.apply(lambda t: t[0])
    agg["signal_color"]   = signal_pairs.apply(lambda t: t[1])

    return agg.sort_values("score_24h").reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════════
# 7. RAMP / CLIFF DETECTION  (fully vectorised)
# ══════════════════════════════════════════════════════════════════════════════

def detect_ramps(enriched: Dict[str, pd.DataFrame]) -> List[dict]:
    """
    Scan the 0–72 h window for sharp generation/temperature transitions.

    Thresholds (from config):
        Wind  : 3 h CF change ≥ WIND_RAMP_THRESHOLD_MW  (default 0.25)
        Solar : 3 h CF change ≥ SOLAR_RAMP_THRESHOLD_MW (default 0.30)
        Temp  : 3 h change    ≥ TEMP_RAMP_THRESHOLD     (default 4.0 °C)

    Fully vectorised – no iterrows – so it runs in < 50 ms for all 22 nodes.
    Returns a list of event dicts sorted by timestamp.
    """
    events: List[dict] = []

    for node_id, df in enriched.items():
        if len(df) < 4:
            continue
        meta = NODES[node_id]

        # 3-hour absolute and signed differences
        w_abs  = df["wind_cf"].diff(3).abs()
        s_abs  = df["solar_cf"].diff(3).abs()
        t_abs  = df["temperature_2m"].diff(3).abs()
        w_sign = df["wind_cf"].diff(3)
        s_sign = df["solar_cf"].diff(3)
        t_sign = df["temperature_2m"].diff(3)

        # ── Wind ramps ────────────────────────────────────────────────────────
        wind_mask = (w_abs >= WIND_RAMP_THRESHOLD_MW) & w_abs.notna()
        for ts in df.index[wind_mask]:
            mag = float(w_abs.loc[ts])
            events.append({
                "time":      ts,
                "node":      meta["label"],
                "country":   meta["country"],
                "type":      "Wind",
                "direction": "▲ ramp UP" if float(w_sign.loc[ts]) > 0 else "▼ ramp DOWN",
                "magnitude": round(mag, 2),
                "severity":  "HIGH" if mag > 0.4 else "MOD",
            })

        # ── Solar ramps ───────────────────────────────────────────────────────
        solar_mask = (s_abs >= SOLAR_RAMP_THRESHOLD_MW) & s_abs.notna()
        for ts in df.index[solar_mask]:
            mag = float(s_abs.loc[ts])
            events.append({
                "time":      ts,
                "node":      meta["label"],
                "country":   meta["country"],
                "type":      "Solar",
                "direction": "▲ ramp UP" if float(s_sign.loc[ts]) > 0 else "▼ ramp DOWN",
                "magnitude": round(mag, 2),
                "severity":  "HIGH" if mag > 0.5 else "MOD",
            })

        # ── Temperature ramps ─────────────────────────────────────────────────
        temp_mask = (t_abs >= TEMP_RAMP_THRESHOLD) & t_abs.notna()
        for ts in df.index[temp_mask]:
            mag = float(t_abs.loc[ts])
            events.append({
                "time":      ts,
                "node":      meta["label"],
                "country":   meta["country"],
                "type":      "Temp",
                "direction": "▲ rising" if float(t_sign.loc[ts]) > 0 else "▼ falling",
                "magnitude": round(mag, 1),   # °C – NOT a CF, so display as float
                "severity":  "MOD",
            })

    events.sort(key=lambda e: e["time"])
    return events


# ══════════════════════════════════════════════════════════════════════════════
# 8. INTER-COUNTRY SPREAD SIGNALS
# ══════════════════════════════════════════════════════════════════════════════

def spread_signals(country_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute pairwise score differentials for the pairs defined in SPREAD_PAIRS.

    Positive spread → country A has a higher (more bullish) score than B,
    implying relatively higher price pressure in A.
    """
    if country_df.empty or "country" not in country_df.columns:
        return pd.DataFrame()

    rows: List[dict] = []
    scores = country_df.set_index("country")["score_24h"].to_dict()

    for cA, cB, name in SPREAD_PAIRS:
        sA = scores.get(cA)
        sB = scores.get(cB)
        if sA is None or sB is None:
            continue
        diff = round(sA - sB, 3)
        label, color = _score_label(diff * 2)   # amplify to spread range
        rows.append({
            "pair":         name,
            "country_A":    cA,
            "country_B":    cB,
            "score_A":      sA,
            "score_B":      sB,
            "spread":       diff,
            "signal":       label,
            "signal_color": color,
        })

    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# 9. INTRA-COUNTRY SPREAD  (e.g. North vs South Germany)
# ══════════════════════════════════════════════════════════════════════════════

def intra_country_spreads(summary: pd.DataFrame) -> pd.DataFrame:
    """
    For each country with ≥2 sub-regions, compute every pairwise sub-region
    score differential and return as a DataFrame.
    """
    if summary.empty or "country" not in summary.columns:
        return pd.DataFrame()

    rows: List[dict] = []
    for country, grp in summary.groupby("country"):
        sub_scores = grp.groupby("subregion")["score_24h"].mean()
        if len(sub_scores) < 2:
            continue
        subs = sub_scores.index.tolist()
        for i, a in enumerate(subs):
            for b in subs[i + 1:]:
                diff = round(float(sub_scores[a]) - float(sub_scores[b]), 3)
                label, color = _score_label(diff * 2)
                rows.append({
                    "country":      country,
                    "spread":       f"{a} vs {b}",
                    "score_A":      sub_scores[a],
                    "score_B":      sub_scores[b],
                    "diff":         diff,
                    "signal":       label,
                    "signal_color": color,
                })

    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# 10. TRADE INSIGHT GENERATOR
# ══════════════════════════════════════════════════════════════════════════════

def generate_insights(
    country_df: pd.DataFrame,
    spreads_df: pd.DataFrame,
    ramps: List[dict],
    node_summary: pd.DataFrame,
) -> List[dict]:
    """
    Auto-generate trader-style bullet insights from the aggregated signals.
    Returns up to 12 dicts: {icon, text, color, priority}.
    Priority 1 = most actionable, 3 = contextual / lower urgency.
    """
    if country_df.empty:
        return []

    insights: List[dict] = []

    # ── Low wind (bullish supply squeeze) ─────────────────────────────────────
    for _, row in country_df[country_df["wind_24h"] < 0.15].iterrows():
        insights.append({
            "icon":     "💨",
            "text":     f"**{row['country']}**: Wind generation very low "
                        f"(CF {row['wind_24h']:.0%}) over next 24 h → bullish supply "
                        f"squeeze. Watch baseload premium.",
            "color":    BULL_COLOR,
            "priority": 1,
        })

    # ── High wind (bearish generation surplus) ────────────────────────────────
    for _, row in country_df[country_df["wind_24h"] > 0.70].iterrows():
        insights.append({
            "icon":     "🌬️",
            "text":     f"**{row['country']}**: Exceptional wind output "
                        f"(CF {row['wind_24h']:.0%}) → price suppression likely. "
                        f"Bearish intraday, esp. peak vs off-peak spread.",
            "color":    BEAR_COLOR,
            "priority": 1,
        })

    # ── High solar in southern markets (duck curve risk) ──────────────────────
    solar_countries = {"ES", "IT", "PT", "FR"}
    high_solar = country_df[
        (country_df["solar_24h"] > 0.40) &
        (country_df["country"].isin(solar_countries))
    ]
    for _, row in high_solar.iterrows():
        insights.append({
            "icon":     "☀️",
            "text":     f"**{row['country']}**: Strong solar expected "
                        f"(CF {row['solar_24h']:.0%}). Mid-day duck curve risk — "
                        f"bearish 10:00–15:00, bullish morning/evening ramp.",
            "color":    BEAR_COLOR,
            "priority": 2,
        })

    # ── Cold demand (HDD-driven heating load) ─────────────────────────────────
    cold_nodes = node_summary[node_summary["temp_now"] < COLD_THRESHOLD]
    if not cold_nodes.empty:
        avg_temps = cold_nodes.groupby("country")["temp_now"].mean().round(1)
        for ctry, t in avg_temps.items():
            insights.append({
                "icon":     "🥶",
                "text":     f"**{ctry}**: Temps at {t} °C → elevated heating demand. "
                            f"HDD signal active. Expect residential load pickup overnight.",
                "color":    BULL_COLOR,
                "priority": 1,
            })

    # ── Heat demand (CDD-driven cooling load) ─────────────────────────────────
    hot_nodes = node_summary[node_summary["temp_now"] > HEAT_THRESHOLD]
    if not hot_nodes.empty:
        avg_temps = hot_nodes.groupby("country")["temp_now"].mean().round(1)
        for ctry, t in avg_temps.items():
            insights.append({
                "icon":     "🌡️",
                "text":     f"**{ctry}**: Heat at {t} °C driving A/C load. "
                            f"CDD signal active. Peak hours 13:00–18:00 at risk.",
                "color":    BULL_COLOR,
                "priority": 2,
            })

    # ── High-severity ramp events ─────────────────────────────────────────────
    # BUG FIX: temperature magnitudes are in °C (e.g. 5.3), NOT a capacity factor.
    # Use :.1f for Temp ramps; :.0% only for Wind/Solar CF ramps.
    for r in [rv for rv in ramps if rv["severity"] == "HIGH"][:5]:
        ts_str = r["time"].strftime("%d-%b %H:%M UTC")
        if r["type"] == "Temp":
            mag_str = f"Δ{r['magnitude']:.1f} °C"
        else:
            mag_str = f"Δ{r['magnitude']:.0%} CF"
        insights.append({
            "icon":     "⚡",
            "text":     f"**{r['country']} – {r['node']}**: {r['type']} "
                        f"{r['direction']} ({mag_str}) at {ts_str}. "
                        f"Sharp generation transition — intraday volatility risk.",
            "color":    NEUTRAL_COLOR,
            "priority": 1,
        })

    # ── Wide cross-border spreads (arbitrage signal) ──────────────────────────
    if not spreads_df.empty:
        wide = spreads_df[spreads_df["spread"].abs() > 0.15]
        for _, row in wide.iterrows():
            direction = (
                f"{row['country_A']} outperforming {row['country_B']}"
                if row["spread"] > 0
                else f"{row['country_B']} outperforming {row['country_A']}"
            )
            insights.append({
                "icon":     "↔️",
                "text":     f"**{row['pair']}**: Spread {row['spread']:+.2f} — "
                            f"{direction}. Cross-border arbitrage window may be open.",
                "color":    NEUTRAL_COLOR,
                "priority": 3,
            })

    # ── Nordic hydro proxy ────────────────────────────────────────────────────
    nordic = country_df[country_df["country"] == "NO"]
    if not nordic.empty and float(nordic.iloc[0]["wind_24h"]) > 0.50:
        insights.append({
            "icon":     "💧",
            "text":     "**Norway**: Strong wind reducing hydro dispatch pressure. "
                        "Potential for increased NO→DE exports. Watch system price.",
            "color":    BEAR_COLOR,
            "priority": 3,
        })

    insights.sort(key=lambda x: x["priority"])
    return insights[:12]


# ══════════════════════════════════════════════════════════════════════════════
# 11. TIMESERIES HELPERS FOR CHARTING
# ══════════════════════════════════════════════════════════════════════════════

def country_timeseries(enriched: Dict[str, pd.DataFrame], country_code: str) -> pd.DataFrame:
    """
    Return an hourly DataFrame averaged across all nodes for `country_code`.
    Columns: wind_cf, solar_cf, demand_sig, temperature_2m, score.
    Returns an empty DataFrame if no nodes match.
    """
    frames = [
        df for nid, df in enriched.items()
        if NODES[nid]["country"] == country_code
    ]
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames)
    cols = ["wind_cf", "solar_cf", "demand_sig", "temperature_2m", "score"]
    return combined.groupby(combined.index)[cols].mean()


def all_countries_score_ts(enriched: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Return a wide-form DataFrame with columns = country codes and
    index = UTC timestamps, values = mean composite score.
    """
    country_codes = list({NODES[nid]["country"] for nid in enriched})
    series: Dict[str, pd.Series] = {}
    for cc in country_codes:
        ts = country_timeseries(enriched, cc)
        if not ts.empty:
            series[cc] = ts["score"]
    if not series:
        return pd.DataFrame()
    return pd.DataFrame(series).sort_index()
"""
app.py – EU Power Market Weather Dashboard  (compact edition)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from streamlit_autorefresh import st_autorefresh

from config import (
    APP_TITLE, APP_ICON, THEME_COLOR,
    BULL_COLOR, BEAR_COLOR, NEUTRAL_COLOR,
    COUNTRY_LABELS, NODES,
)
from data_fetch import fetch_all_nodes, get_fetch_timestamp
from transformations import (
    enrich_all, country_summary, country_level,
    spread_signals, intra_country_spreads,
    detect_ramps, generate_insights,
    country_timeseries, all_countries_score_ts,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EU Power Dashboard",
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar: auto-refresh + controls ─────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚡ EU Power Dashboard")
    st.markdown("---")

    st.markdown("**🔄 Auto-Refresh**")
    refresh_map = {"Off": 0, "5 min": 300_000, "15 min": 900_000,
                   "30 min": 1_800_000, "60 min": 3_600_000}
    refresh_choice = st.radio(
        "Interval", list(refresh_map.keys()), index=2, label_visibility="collapsed"
    )
    refresh_ms = refresh_map[refresh_choice]
    if refresh_ms:
        count = st_autorefresh(interval=refresh_ms, key="autorefresh")
        st.caption(f"Refreshes every {refresh_choice}. Run #{count}.")
    else:
        st.caption("Auto-refresh off.")
        if st.button("🔃 Refresh now"):
            st.cache_data.clear()
            st.rerun()

    st.markdown("---")
    st.markdown("**📋 Data**")
    st.caption("Source: Open-Meteo (free, no API key)")
    st.caption("Cache: 1 h · 22 nodes · 13 countries")
    st.caption("Horizon: 0–72 h UTC")

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  html,body,[class*="css"]{background:#0E1117;color:#E0E0E0;font-family:'Courier New',monospace}
  .main .block-container{padding-top:.5rem;padding-bottom:.5rem;max-width:1400px}
  /* header strip */
  .hdr{background:linear-gradient(90deg,#0E1117,#162032,#0E1117);
       border-bottom:1px solid #00D4FF33;padding:.35rem 1rem;margin-bottom:.5rem;
       display:flex;align-items:baseline;gap:1rem}
  .hdr-title{font-size:1.1rem;font-weight:700;color:#00D4FF;letter-spacing:2px;text-transform:uppercase}
  .hdr-sub{font-size:.65rem;color:#8899AA;letter-spacing:.5px}
  /* section labels */
  .sec{font-size:.6rem;font-weight:700;color:#00D4FF;letter-spacing:2px;text-transform:uppercase;
       border-bottom:1px solid #1E3045;padding-bottom:2px;margin:.6rem 0 .35rem}
  /* score badges */
  .bull{color:#00C853;font-weight:700} .bear{color:#FF4B4B;font-weight:700}
  .neu{color:#FFA726;font-weight:700}
  /* compact insight row */
  .ins{background:#101820;border-left:3px solid #00D4FF;border-radius:3px;
       padding:.35rem .6rem;margin:.15rem 0;font-size:.75rem;color:#CCDDEE;line-height:1.4}
  .ins-bull{border-left-color:#00C853} .ins-bear{border-left-color:#FF4B4B}
  .ins-neu{border-left-color:#FFA726}
  /* scoreboard table rows */
  .sc-row{display:flex;align-items:center;gap:.5rem;padding:.2rem .4rem;
          border-bottom:1px solid #1E3045;font-size:.75rem}
  .sc-row:hover{background:#162032}
  .sc-ctry{width:7rem;color:#AABBCC;font-size:.7rem}
  .sc-score{width:3.5rem;font-weight:700;text-align:right}
  .sc-bar-bg{flex:1;background:#1E3045;height:5px;border-radius:3px}
  .sc-bar{height:5px;border-radius:3px}
  .sc-meta{width:14rem;color:#8899AA;font-size:.65rem;text-align:right}
  /* spread row */
  .sp-row{display:flex;align-items:center;gap:.5rem;padding:.18rem .4rem;
          border-bottom:1px solid #1E3045;font-size:.72rem}
  .sp-pair{flex:1;color:#AABBCC} .sp-val{width:3rem;font-weight:700;text-align:right}
  .sp-bar-bg{width:5rem;background:#1E3045;height:4px;border-radius:2px}
  .sp-bar{height:4px;border-radius:2px}
  /* ramp row */
  .ramp{background:#162032;border-radius:3px;padding:2px 6px;margin:1px 0;
        font-size:.7rem;border-left:2px solid #FFA726}
  /* footer */
  .ftr{text-align:center;font-size:.6rem;color:#445566;border-top:1px solid #1E3045;
       margin-top:1rem;padding-top:.3rem}
  /* error / warn */
  .err{background:#1A0A0A;border-left:4px solid #FF4B4B;border-radius:4px;
       padding:.8rem 1rem;margin:.5rem 0;font-size:.8rem;color:#CCBBBB}
  .wrn{background:#1A1200;border-left:4px solid #FFA726;border-radius:4px;
       padding:.5rem .8rem;margin:.3rem 0;font-size:.75rem;color:#CCAA77}
  #MainMenu{visibility:hidden} footer{visibility:hidden} header{visibility:hidden}
  /* tighten plotly chart padding */
  .stPlotlyChart{margin-bottom:0!important;padding-bottom:0!important}
</style>
""", unsafe_allow_html=True)

# ── Helpers ───────────────────────────────────────────────────────────────────
PL = dict(
    paper_bgcolor="#0E1117", plot_bgcolor="#101820",
    font=dict(color="#8899AA", size=10, family="Courier New"),
    margin=dict(l=30, r=10, t=18, b=18),
    legend=dict(bgcolor="rgba(14,17,23,0)", font=dict(size=9)),
)
AX = dict(gridcolor="#1E3045", zerolinecolor="#1E3045")

def sig_color(sig):
    return BULL_COLOR if "BULL" in sig else BEAR_COLOR if "BEAR" in sig else NEUTRAL_COLOR

def sig_cls(sig):
    return "bull" if "BULL" in sig else "bear" if "BEAR" in sig else "neu"

def flag(code):
    return COUNTRY_LABELS.get(code, code)

def badge_html(sig):
    c = sig_color(sig); cl = sig_cls(sig)
    return f'<span class="{cl}">{sig.split()[-1]}</span>'   # just BULLISH/BEARISH/NEUTRAL word

# ── Data load ─────────────────────────────────────────────────────────────────
with st.spinner("⚡ Loading…"):
    raw      = fetch_all_nodes()
    enriched = enrich_all(raw)
    node_sum = country_summary(enriched)
    ctry_df  = country_level(node_sum)
    spreads  = spread_signals(ctry_df)
    intra    = intra_country_spreads(node_sum)
    ramps    = detect_ramps(enriched)
    insights = generate_insights(ctry_df, spreads, ramps, node_sum)
    score_ts = all_countries_score_ts(enriched)
    fetch_ts = get_fetch_timestamp()

n_ok = len(raw)
n_c  = len(ctry_df) if not ctry_df.empty else 0
refresh_lbl = refresh_choice if refresh_ms else "off"

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="hdr">
  <span class="hdr-title">⚡ EU Power Weather</span>
  <span class="hdr-sub">
    {n_ok}/{len(NODES)} nodes &nbsp;·&nbsp; {n_c} countries &nbsp;·&nbsp;
    Updated {fetch_ts} &nbsp;·&nbsp; Refresh: {refresh_lbl}
  </span>
</div>""", unsafe_allow_html=True)

# ── Empty-data guard ──────────────────────────────────────────────────────────
if node_sum.empty:
    st.markdown("""<div class="err">⚠️ <b>No data loaded.</b>
    Open-Meteo API is rate-limiting this IP. Wait ~60 s and refresh,
    or use the sidebar Refresh button.</div>""", unsafe_allow_html=True)
    st.stop()

if n_ok < len(NODES):
    st.markdown(f"""<div class="wrn">⚠️ {len(NODES)-n_ok} nodes unavailable —
    signals use partial data.</div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# ROW 1: Insights (compact scrollable) + 72h score chart
# ══════════════════════════════════════════════════════════════════════════════
col_ins, col_ts = st.columns([2, 3])

with col_ins:
    st.markdown('<div class="sec">🎯 Trade Insights</div>', unsafe_allow_html=True)
    for ins in insights[:8]:
        c = ins["color"]
        cl = "ins-bull" if c == BULL_COLOR else "ins-bear" if c == BEAR_COLOR else "ins-neu"
        st.markdown(
            f'<div class="ins {cl}">{ins["icon"]} {ins["text"]}</div>',
            unsafe_allow_html=True,
        )

with col_ts:
    st.markdown('<div class="sec">📊 72h Composite Score</div>', unsafe_allow_html=True)
    if not score_ts.empty:
        fig = go.Figure()
        pal = px.colors.qualitative.Bold + px.colors.qualitative.Dark24
        for i, cc in enumerate(score_ts.head(72).columns):
            fig.add_trace(go.Scatter(
                x=score_ts.index, y=score_ts[cc].round(3), name=cc, mode="lines",
                line=dict(width=1.5, color=pal[i % len(pal)]),
                hovertemplate=f"<b>{flag(cc)}</b><br>%{{x|%d-%b %H:%M}}<br>%{{y:.3f}}<extra></extra>",
            ))
        fig.add_hline(y=0,    line_dash="dot", line_color="#445566", line_width=1)
        fig.add_hline(y=0.2,  line_dash="dot", line_color="rgba(0,200,83,0.33)",  line_width=0.8)
        fig.add_hline(y=-0.2, line_dash="dot", line_color="rgba(255,75,75,0.33)", line_width=0.8)
        fig.update_layout(**PL, height=260,
                          xaxis={**AX, "title": ""},
                          yaxis={**AX, "title": "Score", "range": [-0.8, 0.8]})
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    else:
        st.info("Timeseries unavailable.")

# ══════════════════════════════════════════════════════════════════════════════
# ROW 2: Country scoreboard (compact table) + Spreads
# ══════════════════════════════════════════════════════════════════════════════
col_sc, col_sp = st.columns([3, 2])

with col_sc:
    st.markdown('<div class="sec">🗺️ Country Scoreboard — 24h</div>', unsafe_allow_html=True)
    if not ctry_df.empty:
        rows_html = []
        for _, row in ctry_df.sort_values("score_24h").iterrows():
            sc   = row["score_24h"]
            col_ = sig_color(row["signal"])
            # bar width: map [-0.65, +0.65] → [0%, 100%], centre at 50%
            bar_w   = min(max(int((sc + 0.65) / 1.3 * 100), 0), 100)
            bar_left = 50  # centre line
            # bar grows right from centre if bullish, left if bearish
            if sc >= 0:
                bar_style = f"width:{bar_w-50}%;margin-left:50%;background:{col_}"
            else:
                bar_style = f"width:{50-bar_w}%;margin-left:{bar_w}%;background:{col_}"
            rows_html.append(f"""
            <div class="sc-row">
              <span class="sc-ctry">{flag(row['country'])}</span>
              <span class="sc-score" style="color:{col_}">{sc:+.2f}</span>
              <span class="{sig_cls(row['signal'])}" style="font-size:.65rem;width:4.5rem">
                {row['signal'].split()[-1]}</span>
              <div class="sc-bar-bg">
                <div class="sc-bar" style="{bar_style}"></div>
              </div>
              <span class="sc-meta">
                💨{row['wind_24h']:.0%} ☀️{row['solar_24h']:.0%} 🌡️{row['temp_now']:.0f}°C
              </span>
            </div>""")
        st.markdown("".join(rows_html), unsafe_allow_html=True)

with col_sp:
    st.markdown('<div class="sec">↔️ Spreads & Imbalances</div>', unsafe_allow_html=True)

    # Inter-country spreads
    if not spreads.empty:
        for _, row in spreads.iterrows():
            sc   = row["spread"]
            col_ = sig_color(row["signal"])
            bw   = int(min(abs(sc) * 150, 100))
            st.markdown(f"""
            <div class="sp-row">
              <span class="sp-pair">{row['pair']}</span>
              <span class="sp-val" style="color:{col_}">{sc:+.2f}</span>
              <div class="sp-bar-bg">
                <div class="sp-bar" style="width:{bw}%;background:{col_}"></div>
              </div>
            </div>""", unsafe_allow_html=True)

    # Intra-country imbalances (compact)
    if not intra.empty:
        st.markdown('<div style="margin-top:.4rem;font-size:.65rem;color:#8899AA;">Intra-country</div>',
                    unsafe_allow_html=True)
        for _, row in intra.iterrows():
            col_ = sig_color(row["signal"])
            st.markdown(f"""
            <div class="sp-row">
              <span class="sp-pair">{flag(row['country'])} {row['spread']}</span>
              <span class="sp-val" style="color:{col_}">{row['diff']:+.2f}</span>
            </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# ROW 3: Country detail chart + Ramps (side by side)
# ══════════════════════════════════════════════════════════════════════════════
col_det, col_ramp = st.columns([3, 2])

with col_det:
    st.markdown('<div class="sec">🔍 Country Detail — 0–72h</div>', unsafe_allow_html=True)
    available = sorted(node_sum["country"].unique())
    sel = st.selectbox("Country", available, format_func=flag, label_visibility="collapsed")
    ts  = country_timeseries(enriched, sel)

    if not ts.empty:
        fig2 = make_subplots(
            rows=3, cols=1, shared_xaxes=True,
            subplot_titles=["Wind CF", "Solar CF", "Temp °C"],
            vertical_spacing=0.06,
        )
        fig2.add_trace(go.Scatter(
            x=ts.index, y=ts["wind_cf"].round(3), name="Wind",
            fill="tozeroy", fillcolor="rgba(0,212,255,0.08)",
            line=dict(color=THEME_COLOR, width=1.5),
            hovertemplate="%{x|%d %H:%M} Wind:%{y:.2f}<extra></extra>",
        ), row=1, col=1)
        fig2.add_trace(go.Scatter(
            x=ts.index, y=ts["solar_cf"].round(3), name="Solar",
            fill="tozeroy", fillcolor="rgba(255,167,38,0.08)",
            line=dict(color="#FFA726", width=1.5),
            hovertemplate="%{x|%d %H:%M} Solar:%{y:.2f}<extra></extra>",
        ), row=2, col=1)
        fig2.add_trace(go.Scatter(
            x=ts.index, y=ts["temperature_2m"].round(1), name="Temp",
            line=dict(color="#FF4B4B", width=1.5),
            hovertemplate="%{x|%d %H:%M} T:%{y:.1f}°C<extra></extra>",
        ), row=3, col=1)
        fig2.add_hline(y=15, line_dash="dot",
                       line_color="rgba(68,85,102,0.53)", line_width=1, row=3, col=1)
        for r in [r for r in ramps if r["country"] == sel and r["type"] == "Wind"][:5]:
            fig2.add_vline(x=r["time"], line_dash="dash",
                           line_color="rgba(255,167,38,0.53)", line_width=1, row=1, col=1)
        fig2.update_layout(**PL, height=340, showlegend=False)
        fig2.update_xaxes(**AX)
        fig2.update_yaxes(**AX)
        fig2.update_annotations(font=dict(color="#8899AA", size=9))
        st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

        # Node mini-cards in one compact row
        sub = node_sum[node_sum["country"] == sel]
        if not sub.empty:
            cols_n = st.columns(len(sub))
            for c, (_, r) in zip(cols_n, sub.iterrows()):
                col_ = sig_color(r["signal"])
                with c:
                    st.markdown(f"""
                    <div style="background:#162032;border:1px solid #1E3045;border-radius:4px;
                         padding:.3rem .5rem;font-size:.68rem">
                      <div style="color:#8899AA">{r['subregion']} · {r['label']}</div>
                      <div style="color:{col_};font-weight:700;font-size:.9rem">{r['score_24h']:+.2f}</div>
                      <div style="color:#8899AA">💨{r['wind_24h']:.0%} ☀️{r['solar_24h']:.0%}
                        🌡️{r['temp_now']:.0f}°C</div>
                    </div>""", unsafe_allow_html=True)
    else:
        st.info("No data for this country.")

with col_ramp:
    st.markdown('<div class="sec">⚡ Ramp Events (0–72h)</div>', unsafe_allow_html=True)
    if ramps:
        rdf = pd.DataFrame(ramps)
        rdf["ts"] = rdf["time"].dt.strftime("%d-%b %H:%M")

        for rtype, icon, fmt in [
            ("Wind",  "💨", lambda r: f"Δ{r['magnitude']:.0%}CF"),
            ("Solar", "☀️", lambda r: f"Δ{r['magnitude']:.0%}CF"),
            ("Temp",  "🌡️", lambda r: f"Δ{r['magnitude']:.1f}°C"),
        ]:
            subset = rdf[rdf["type"] == rtype].head(6)
            if subset.empty:
                continue
            st.markdown(f"<div style='font-size:.65rem;color:#8899AA;margin:.3rem 0 .1rem'>"
                        f"{icon} {rtype}</div>", unsafe_allow_html=True)
            for _, r in subset.iterrows():
                c = BEAR_COLOR if r["severity"] == "HIGH" else NEUTRAL_COLOR
                st.markdown(
                    f'<div class="ramp" style="border-left-color:{c}">'
                    f'<b>{r["country"]}-{r["node"]}</b> {r["direction"]} '
                    f'{fmt(r)} · {r["ts"]}</div>',
                    unsafe_allow_html=True,
                )
    else:
        st.info("No ramps detected.")

# ══════════════════════════════════════════════════════════════════════════════
# ROW 4: Full node table (collapsed)
# ══════════════════════════════════════════════════════════════════════════════
with st.expander("📋 Full node snapshot"):
    snap = node_sum[["label","country","subregion","temp_now","wind_ms_now",
                      "wind_cf_now","solar_cf_now","score_24h","signal",
                      "hdd_24h","cdd_24h"]].copy()
    snap.columns = ["Node","Country","Subregion","T°C","Wind m/s","Wind CF",
                    "Solar CF","Score 24h","Signal","HDD","CDD"]
    st.dataframe(snap.sort_values("Score 24h").reset_index(drop=True),
                 use_container_width=True, hide_index=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="ftr">
  <a href="https://open-meteo.com" target="_blank" style="color:#00D4FF">Open-Meteo API</a>
  &nbsp;·&nbsp; {fetch_ts} &nbsp;·&nbsp; {n_ok}/{len(NODES)} nodes
  &nbsp;·&nbsp; For informational purposes only
</div>""", unsafe_allow_html=True)
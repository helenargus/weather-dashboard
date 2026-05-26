"""
app.py – European Power Market Weather Dashboard
Entry point for Streamlit Community Cloud deployment.

Run locally:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from config import (
    APP_TITLE, APP_ICON, THEME_COLOR,
    BULL_COLOR, BEAR_COLOR, NEUTRAL_COLOR,
    COUNTRY_LABELS, COUNTRY_GROUPS, NODES,
)
from data_fetch import fetch_all_nodes, get_fetch_timestamp, combined_dataframe
from transformations import (
    enrich_all,
    country_summary,
    country_level,
    spread_signals,
    intra_country_spreads,
    detect_ramps,
    generate_insights,
    country_timeseries,
    all_countries_score_ts,
)

# ─── Page config (must be first Streamlit call) ───────────────────────────────
st.set_page_config(
    page_title="EU Power Weather Dashboard",
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── Dark-mode CSS ────────────────────────────────────────────────────────────
st.markdown("""
<style>
  html, body, [class*="css"] {
      background-color: #0E1117; color: #E0E0E0;
      font-family: 'Courier New', monospace;
  }
  .main .block-container { padding-top: 1rem; max-width: 1400px; }
  .dash-header {
      background: linear-gradient(90deg,#0E1117 0%,#162032 50%,#0E1117 100%);
      border-bottom: 1px solid #00D4FF33; padding: 0.6rem 1rem; margin-bottom:1rem;
  }
  .dash-title { font-size:1.6rem; font-weight:700; color:#00D4FF;
      letter-spacing:2px; text-transform:uppercase; }
  .dash-subtitle { font-size:0.75rem; color:#8899AA; letter-spacing:1px; }
  .metric-card {
      background:#162032; border:1px solid #1E3045; border-radius:6px;
      padding:0.7rem 0.9rem; margin-bottom:0.5rem;
  }
  .metric-card:hover { border-color:#00D4FF55; }
  .metric-label { font-size:0.65rem; color:#8899AA; text-transform:uppercase; letter-spacing:1px; }
  .metric-value { font-size:1.4rem; font-weight:700; color:#E0E0E0; }
  .metric-sub   { font-size:0.7rem; color:#8899AA; }
  .badge-bull    { background:#00C85322; border:1px solid #00C853; color:#00C853;
      padding:2px 10px; border-radius:3px; font-size:0.75rem; font-weight:700; }
  .badge-bear    { background:#FF4B4B22; border:1px solid #FF4B4B; color:#FF4B4B;
      padding:2px 10px; border-radius:3px; font-size:0.75rem; font-weight:700; }
  .badge-neutral { background:#FFA72622; border:1px solid #FFA726; color:#FFA726;
      padding:2px 10px; border-radius:3px; font-size:0.75rem; font-weight:700; }
  .insight-panel {
      background:#101820; border:1px solid #00D4FF33; border-left:3px solid #00D4FF;
      border-radius:6px; padding:0.8rem 1rem; margin-bottom:0.5rem;
  }
  .insight-bull    { border-left-color:#00C853; }
  .insight-bear    { border-left-color:#FF4B4B; }
  .insight-neutral { border-left-color:#FFA726; }
  .insight-text    { font-size:0.82rem; color:#CCDDEE; line-height:1.5; }
  .insight-icon    { font-size:1.1rem; margin-right:6px; }
  .section-header {
      font-size:0.7rem; font-weight:700; color:#00D4FF; letter-spacing:2px;
      text-transform:uppercase; border-bottom:1px solid #1E3045;
      padding-bottom:4px; margin:1rem 0 0.6rem 0;
  }
  .ramp-row { background:#162032; border-radius:4px; padding:4px 8px; margin:2px 0;
      font-size:0.78rem; border-left:3px solid #FFA726; }
  .dash-footer {
      text-align:center; font-size:0.65rem; color:#445566;
      border-top:1px solid #1E3045; margin-top:2rem; padding-top:0.5rem; letter-spacing:1px;
  }
  .error-banner {
      background:#1A0A0A; border:1px solid #FF4B4B55; border-left:4px solid #FF4B4B;
      border-radius:6px; padding:1.2rem 1.5rem; margin:1rem 0;
  }
  .error-banner h3 { color:#FF4B4B; margin:0 0 0.5rem 0; font-size:1rem; }
  .error-banner p  { color:#CCBBBB; margin:0; font-size:0.85rem; line-height:1.6; }
  .warn-banner {
      background:#1A1200; border:1px solid #FFA72655; border-left:4px solid #FFA726;
      border-radius:6px; padding:0.8rem 1.2rem; margin:0.5rem 0;
  }
  .warn-banner p { color:#CCAA77; margin:0; font-size:0.82rem; }
  #MainMenu { visibility:hidden; }
  footer    { visibility:hidden; }
  header    { visibility:hidden; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

# Base Plotly layout: intentionally NO 'xaxis'/'yaxis' keys so per-chart
# overrides don't trigger "multiple values for keyword argument" TypeError.
PLOTLY_LAYOUT = dict(
    paper_bgcolor="#0E1117",
    plot_bgcolor="#101820",
    font=dict(color="#8899AA", size=11, family="Courier New"),
    margin=dict(l=40, r=20, t=30, b=30),
    legend=dict(bgcolor="rgba(14,17,23,0)", font=dict(size=10)),
)
_AXIS_STYLE = dict(gridcolor="#1E3045", zerolinecolor="#1E3045")


def badge(signal: str) -> str:
    cls = ("badge-bull"    if "BULL" in signal else
           "badge-bear"    if "BEAR" in signal else "badge-neutral")
    return f'<span class="{cls}">{signal}</span>'


def signal_color(signal: str) -> str:
    return (BULL_COLOR    if "BULL" in signal else
            BEAR_COLOR    if "BEAR" in signal else NEUTRAL_COLOR)


def country_flag(code: str) -> str:
    return COUNTRY_LABELS.get(code, code)


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOAD
# ══════════════════════════════════════════════════════════════════════════════

with st.spinner("⚡ Loading European weather data…"):
    raw_data  = fetch_all_nodes()
    enriched  = enrich_all(raw_data)
    node_sum  = country_summary(enriched)
    ctry_df   = country_level(node_sum)
    spreads   = spread_signals(ctry_df)
    intra     = intra_country_spreads(node_sum)
    ramps     = detect_ramps(enriched)
    insights  = generate_insights(ctry_df, spreads, ramps, node_sum)
    score_ts  = all_countries_score_ts(enriched)
    fetch_ts  = get_fetch_timestamp()

# ── Counts for header display ─────────────────────────────────────────────────
n_nodes_loaded   = len(raw_data)
n_countries_loaded = len(ctry_df) if not ctry_df.empty else 0


# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════

st.markdown(f"""
<div class="dash-header">
  <div class="dash-title">{APP_TITLE}</div>
  <div class="dash-subtitle">
    0–72h &nbsp;•&nbsp; {n_nodes_loaded}/{len(NODES)} Nodes &nbsp;•&nbsp;
    {n_countries_loaded} Countries &nbsp;•&nbsp;
    Open-Meteo API &nbsp;•&nbsp; Updated: {fetch_ts}
  </div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# EARLY-EXIT GUARD — show a clear error and stop if ALL data failed to load
# ══════════════════════════════════════════════════════════════════════════════

if node_sum.empty:
    st.markdown("""
    <div class="error-banner">
      <h3>⚠️ Weather data unavailable</h3>
      <p>
        The Open-Meteo API did not return data for any node on this load.<br>
        This usually means the free API is temporarily overloaded or rate-limiting
        Streamlit Cloud's shared IP address.<br><br>
        <strong>What to do:</strong><br>
        • Wait 60–90 seconds and refresh the page — the cached result will be retried.<br>
        • The 1-hour cache means this banner will only appear once per cache cycle.<br>
        • If the problem persists, check
          <a href="https://open-meteo.com" target="_blank" style="color:#FF8888;">open-meteo.com</a>
          for service status.
      </p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()   # halt rendering — nothing below will execute


# ── Partial-load warning (some nodes returned, not all) ──────────────────────
if n_nodes_loaded < len(NODES):
    missing = len(NODES) - n_nodes_loaded
    st.markdown(f"""
    <div class="warn-banner">
      <p>⚠️ <strong>{missing} of {len(NODES)} nodes</strong> could not be fetched this cycle.
      Signals are computed from available data only. Refresh in ~60s to retry.</p>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TRADE INSIGHT PANEL
# ══════════════════════════════════════════════════════════════════════════════

st.markdown('<div class="section-header">🎯 Trade Insight Panel — Auto-Generated Signals</div>',
            unsafe_allow_html=True)

if insights:
    cols_per_row = 2
    for i in range(0, len(insights), cols_per_row):
        row_insights = insights[i : i + cols_per_row]
        cols = st.columns(cols_per_row)
        for col, ins in zip(cols, row_insights):
            color = ins["color"]
            cls   = ("insight-bull"    if color == BULL_COLOR   else
                     "insight-bear"    if color == BEAR_COLOR   else "insight-neutral")
            with col:
                st.markdown(f"""
                <div class="insight-panel {cls}">
                  <span class="insight-icon">{ins['icon']}</span>
                  <span class="insight-text">{ins['text']}</span>
                </div>""", unsafe_allow_html=True)
else:
    st.info("No significant signals detected at this time.")


# ══════════════════════════════════════════════════════════════════════════════
# COUNTRY SCOREBOARD
# ══════════════════════════════════════════════════════════════════════════════

st.markdown('<div class="section-header">🗺️ Country Scoreboard — 24h Outlook</div>',
            unsafe_allow_html=True)

if not ctry_df.empty:
    n_cols      = 4
    sorted_ctry = ctry_df.sort_values("score_24h").reset_index(drop=True)
    cols        = st.columns(n_cols)
    for idx, row in sorted_ctry.iterrows():
        col = cols[idx % n_cols]
        sc  = row["score_24h"]
        bar_color = signal_color(row["signal"])
        with col:
            st.markdown(f"""
            <div class="metric-card">
              <div class="metric-label">{country_flag(row['country'])}</div>
              <div class="metric-value" style="color:{bar_color};">{sc:+.2f}</div>
              <div style="margin:4px 0;">{badge(row['signal'])}</div>
              <div class="metric-sub">
                🌬️ Wind CF {row['wind_24h']:.0%} &nbsp;
                ☀️ Solar {row['solar_24h']:.0%}<br>
                🌡️ Temp {row['temp_now']:.1f}°C &nbsp;
                Demand {row['demand_24h']:+.2f}
              </div>
            </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# 72h SCORE TIMESERIES
# ══════════════════════════════════════════════════════════════════════════════

st.markdown('<div class="section-header">📊 72h Score Evolution — All Countries</div>',
            unsafe_allow_html=True)

if not score_ts.empty:
    ts_plot = score_ts.head(72)
    fig     = go.Figure()
    palette = px.colors.qualitative.Bold + px.colors.qualitative.Dark24
    for i, cc in enumerate(ts_plot.columns):
        fig.add_trace(go.Scatter(
            x=ts_plot.index, y=ts_plot[cc].round(3), name=cc,
            mode="lines",
            line=dict(width=1.5, color=palette[i % len(palette)]),
            hovertemplate=(f"<b>{country_flag(cc)}</b><br>"
                           f"%{{x|%d-%b %H:%M}}<br>Score: %{{y:.3f}}<extra></extra>"),
        ))
    fig.add_hline(y=0,    line_dash="dot", line_color="#445566",   line_width=1)
    fig.add_hline(y=0.2,  line_dash="dot", line_color="#00C85355", line_width=0.8)
    fig.add_hline(y=-0.2, line_dash="dot", line_color="#FF4B4B55", line_width=0.8)
    fig.update_layout(
        **PLOTLY_LAYOUT, height=320,
        xaxis={**_AXIS_STYLE, "title": ""},
        yaxis={**_AXIS_STYLE, "title": "Composite Score", "range": [-0.8, 0.8]},
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
else:
    st.info("Score timeseries not available.")


# ══════════════════════════════════════════════════════════════════════════════
# COUNTRY DETAIL DRILL-DOWN
# ══════════════════════════════════════════════════════════════════════════════

st.markdown('<div class="section-header">🔍 Country Detail — 0–72h Intraday Charts</div>',
            unsafe_allow_html=True)

# Guard: node_sum guaranteed non-empty past the early-exit above
available_countries = sorted(node_sum["country"].unique())

if not available_countries:
    st.info("No country data available for drill-down.")
else:
    selected_country = st.selectbox(
        "Select country for detail view:",
        options=available_countries,
        format_func=country_flag,
        label_visibility="collapsed",
    )

    ts_detail  = country_timeseries(enriched, selected_country)
    sub_nodes  = node_sum[node_sum["country"] == selected_country]

    if not ts_detail.empty:
        fig2 = make_subplots(
            rows=3, cols=1, shared_xaxes=True,
            subplot_titles=["Wind CF (0–1)", "Solar CF (0–1)", "Temperature (°C)"],
            vertical_spacing=0.08,
        )
        fig2.add_trace(go.Scatter(
            x=ts_detail.index, y=ts_detail["wind_cf"].round(3),
            fill="tozeroy", fillcolor="#00D4FF15",
            line=dict(color=THEME_COLOR, width=2), name="Wind CF",
            hovertemplate="%{x|%d-%b %H:%M}<br>Wind CF: %{y:.2f}<extra></extra>",
        ), row=1, col=1)
        fig2.add_trace(go.Scatter(
            x=ts_detail.index, y=ts_detail["solar_cf"].round(3),
            fill="tozeroy", fillcolor="#FFA72615",
            line=dict(color="#FFA726", width=2), name="Solar CF",
            hovertemplate="%{x|%d-%b %H:%M}<br>Solar CF: %{y:.2f}<extra></extra>",
        ), row=2, col=1)
        fig2.add_trace(go.Scatter(
            x=ts_detail.index, y=ts_detail["temperature_2m"].round(1),
            line=dict(color="#FF4B4B", width=2), name="Temp °C",
            hovertemplate="%{x|%d-%b %H:%M}<br>Temp: %{y:.1f}°C<extra></extra>",
        ), row=3, col=1)
        fig2.add_hline(y=15, line_dash="dot", line_color="#44556688",
                       line_width=1, row=3, col=1)
        country_ramps = [r for r in ramps
                         if r["country"] == selected_country and r["type"] == "Wind"]
        for r in country_ramps[:8]:
            fig2.add_vline(x=r["time"], line_dash="dash",
                           line_color="#FFA72688", line_width=1, row=1, col=1)
        fig2.update_layout(**PLOTLY_LAYOUT, height=480, showlegend=False)
        fig2.update_xaxes(**_AXIS_STYLE)
        fig2.update_yaxes(**_AXIS_STYLE)
        fig2.update_annotations(font=dict(color="#8899AA", size=11))
        st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})
    else:
        st.info(f"No timeseries data available for {country_flag(selected_country)}.")

    if not sub_nodes.empty:
        st.markdown(f"**Node breakdown — {country_flag(selected_country)}**")
        cols = st.columns(len(sub_nodes))
        for col, (_, row) in zip(cols, sub_nodes.iterrows()):
            with col:
                sc        = row["score_24h"]
                bar_color = signal_color(row["signal"])
                st.markdown(f"""
                <div class="metric-card">
                  <div class="metric-label">{row['subregion']} · {row['label']}</div>
                  <div class="metric-value" style="color:{bar_color};">{sc:+.2f}</div>
                  <div style="margin:3px 0;">{badge(row['signal'])}</div>
                  <div class="metric-sub">
                    Wind {row['wind_24h']:.0%} · Solar {row['solar_24h']:.0%}<br>
                    T={row['temp_now']:.1f}°C · HDD {row['hdd_24h']:.1f} · CDD {row['cdd_24h']:.1f}
                  </div>
                </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SPREAD SIGNALS
# ══════════════════════════════════════════════════════════════════════════════

st.markdown('<div class="section-header">↔️ Inter-Regional Spread Signals</div>',
            unsafe_allow_html=True)

col_sp1, col_sp2 = st.columns([3, 2])

with col_sp1:
    if not spreads.empty:
        for _, row in spreads.iterrows():
            sc        = row["spread"]
            bar_color = signal_color(row["signal"])
            bar_pct   = min(abs(sc) * 2, 1.0)
            bar_w     = f"{bar_pct * 100:.0f}%"
            bar_dir   = (f"{row['country_A']} stronger" if sc > 0
                         else f"{row['country_B']} stronger")
            st.markdown(f"""
            <div class="metric-card" style="margin-bottom:6px;">
              <div style="display:flex;justify-content:space-between;align-items:center;">
                <span style="font-size:0.8rem;color:#CCDDEE;font-weight:600;">{row['pair']}</span>
                <span style="font-size:0.8rem;color:{bar_color};font-weight:700;">{sc:+.3f}</span>
                {badge(row['signal'])}
              </div>
              <div style="background:#1E3045;height:4px;border-radius:2px;margin-top:6px;">
                <div style="background:{bar_color};width:{bar_w};height:4px;border-radius:2px;"></div>
              </div>
              <div class="metric-sub" style="margin-top:4px;">{bar_dir}</div>
            </div>""", unsafe_allow_html=True)
    else:
        st.markdown('<div class="metric-sub">Spread data unavailable.</div>',
                    unsafe_allow_html=True)

with col_sp2:
    st.markdown("**Intra-Country Imbalances**")
    if not intra.empty:
        for _, row in intra.iterrows():
            diff      = row["diff"]
            bar_color = signal_color(row["signal"])
            st.markdown(f"""
            <div class="metric-card" style="margin-bottom:6px;">
              <div style="font-size:0.75rem;color:#CCDDEE;">
                <b>{country_flag(row['country'])}</b> · {row['spread']}
              </div>
              <div style="font-size:1rem;color:{bar_color};font-weight:700;">{diff:+.3f}</div>
              <div style="margin-top:2px;">{badge(row['signal'])}</div>
            </div>""", unsafe_allow_html=True)
    else:
        st.markdown('<div class="metric-sub">No intra-country spread data.</div>',
                    unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# RAMP / CLIFF EVENTS
# ══════════════════════════════════════════════════════════════════════════════

st.markdown('<div class="section-header">⚡ Ramp & Cliff Events (0–72h)</div>',
            unsafe_allow_html=True)

if ramps:
    ramp_df            = pd.DataFrame(ramps).head(30)
    ramp_df["time_str"] = ramp_df["time"].dt.strftime("%d-%b %H:%M UTC")
    col_r1, col_r2, col_r3 = st.columns(3)

    with col_r1:
        st.markdown("**🌬️ Wind Ramps**")
        for _, r in ramp_df[ramp_df["type"] == "Wind"].head(10).iterrows():
            sev_color = BEAR_COLOR if r["severity"] == "HIGH" else NEUTRAL_COLOR
            st.markdown(f"""
            <div class="ramp-row" style="border-left-color:{sev_color};">
              <b>{r['country']}-{r['node']}</b> {r['direction']}<br>
              <span style="color:#8899AA;">Δ{r['magnitude']:.0%} CF · {r['time_str']}</span>
            </div>""", unsafe_allow_html=True)

    with col_r2:
        st.markdown("**☀️ Solar Ramps**")
        for _, r in ramp_df[ramp_df["type"] == "Solar"].head(10).iterrows():
            st.markdown(f"""
            <div class="ramp-row" style="border-left-color:{NEUTRAL_COLOR};">
              <b>{r['country']}-{r['node']}</b> {r['direction']}<br>
              <span style="color:#8899AA;">Δ{r['magnitude']:.0%} CF · {r['time_str']}</span>
            </div>""", unsafe_allow_html=True)

    with col_r3:
        st.markdown("**🌡️ Temperature Ramps**")
        for _, r in ramp_df[ramp_df["type"] == "Temp"].head(10).iterrows():
            st.markdown(f"""
            <div class="ramp-row" style="border-left-color:{BULL_COLOR};">
              <b>{r['country']}-{r['node']}</b> {r['direction']}<br>
              <span style="color:#8899AA;">Δ{r['magnitude']:.1f}°C · {r['time_str']}</span>
            </div>""", unsafe_allow_html=True)
else:
    st.info("No significant ramp events detected in the 72h window.")


# ══════════════════════════════════════════════════════════════════════════════
# RAW DATA TABLE (collapsed expander — safe: node_sum non-empty past guard)
# ══════════════════════════════════════════════════════════════════════════════

with st.expander("📋 Full Node Data Table (current hour snapshot)"):
    snap = node_sum[[
        "label", "country", "subregion",
        "temp_now", "wind_ms_now", "wind_cf_now", "solar_cf_now",
        "score_24h", "signal", "hdd_24h", "cdd_24h",
    ]].copy()
    snap.columns = [
        "Node", "Country", "Subregion",
        "Temp °C", "Wind m/s", "Wind CF", "Solar CF",
        "Score 24h", "Signal", "HDD 24h", "CDD 24h",
    ]
    st.dataframe(snap.sort_values("Score 24h").reset_index(drop=True),
                 use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════════════════════

st.markdown(f"""
<div class="dash-footer">
  Data: <a href="https://open-meteo.com" target="_blank" style="color:#00D4FF;">
  Open-Meteo API</a> (free, no key required) &nbsp;|&nbsp;
  Last updated: {fetch_ts} &nbsp;|&nbsp; Cache TTL: 1h &nbsp;|&nbsp;
  {n_nodes_loaded}/{len(NODES)} nodes · {n_countries_loaded} countries &nbsp;|&nbsp;
  ⚡ EU Power Weather Dashboard — for informational purposes only
</div>
""", unsafe_allow_html=True)
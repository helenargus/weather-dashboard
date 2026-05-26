"""
data_fetch.py – Fetch hourly forecasts from Open-Meteo for all nodes.

Uses st.cache_data so repeated page renders don't re-hit the API.
Cache TTL = 1 hour (configurable in config.py).
"""

import time
import logging
from datetime import datetime, timezone
from typing import Dict, Optional

import httpx
import pandas as pd
import streamlit as st

from config import (
    NODES,
    OPEN_METEO_URL,
    OPEN_METEO_VARIABLES,
    FORECAST_HOURS,
    CACHE_TTL_SECONDS,
)

logger = logging.getLogger(__name__)

# ─── Single-node fetch (not cached – wrapped below) ───────────────────────────

def _fetch_node(node_id: str, meta: dict, client: httpx.Client) -> Optional[pd.DataFrame]:
    """Fetch one node from Open-Meteo; return tidy DataFrame or None on error."""
    params = {
        "latitude":  meta["lat"],
        "longitude": meta["lon"],
        "hourly":    ",".join(OPEN_METEO_VARIABLES),
        "forecast_days": (FORECAST_HOURS // 24) + 1,
        "timezone":  "UTC",
    }
    try:
        resp = client.get(OPEN_METEO_URL, params=params, timeout=10)
        resp.raise_for_status()
        raw = resp.json()
        hourly = raw.get("hourly", {})
        if not hourly:
            logger.warning("Empty hourly payload for %s", node_id)
            return None

        df = pd.DataFrame(hourly)
        df["time"] = pd.to_datetime(df["time"], utc=True)
        df = df.set_index("time").sort_index()

        # Trim to exactly FORECAST_HOURS
        now_utc = pd.Timestamp.now(tz="UTC").floor("h")
        df = df[df.index >= now_utc].head(FORECAST_HOURS)

        # Attach metadata
        df["node_id"]   = node_id
        df["country"]   = meta["country"]
        df["subregion"] = meta["subregion"]
        df["label"]     = meta["label"]
        return df

    except Exception as exc:
        logger.error("Failed to fetch %s: %s", node_id, exc)
        return None


# ─── Bulk fetch – cached ──────────────────────────────────────────────────────

@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def fetch_all_nodes() -> Dict[str, pd.DataFrame]:
    """
    Return dict { node_id -> DataFrame } for all nodes.
    Cached for CACHE_TTL_SECONDS so the app stays fast.

    Example API call (what this function generates per node):
        GET https://api.open-meteo.com/v1/forecast
            ?latitude=52.52
            &longitude=13.40
            &hourly=temperature_2m,windspeed_10m,windspeed_100m,
                    winddirection_10m,shortwave_radiation,cloudcover,
                    precipitation,surface_pressure
            &forecast_days=4
            &timezone=UTC
    """
    results: Dict[str, pd.DataFrame] = {}
    t0 = time.time()

    # Use a single connection pool for all requests
    with httpx.Client() as client:
        for node_id, meta in NODES.items():
            df = _fetch_node(node_id, meta, client)
            if df is not None:
                results[node_id] = df
            # Polite micro-delay to avoid hammering the free API
            time.sleep(0.05)

    elapsed = time.time() - t0
    logger.info("Fetched %d/%d nodes in %.1fs", len(results), len(NODES), elapsed)
    return results


def get_fetch_timestamp() -> str:
    """Return a human-readable UTC timestamp of when data was last fetched."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


# ─── Convenience: combine all nodes into one long DataFrame ───────────────────

def combined_dataframe(node_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Stack all node DataFrames into a single long-form DataFrame."""
    if not node_data:
        return pd.DataFrame()
    frames = list(node_data.values())
    return pd.concat(frames).sort_index()

"""
data_fetch.py – Fetch hourly forecasts from Open-Meteo for all nodes.

FETCHING STRATEGY (why not batch?)
────────────────────────────────────────────────────────────────────────────────
Open-Meteo's free tier is documented to support multi-location batch queries,
but in practice the shared IPs used by Streamlit Community Cloud trigger
aggressive rate-limiting and mid-request disconnects when all 22 lat/lon pairs
are sent in one request URL.

The reliable approach on shared hosting:
  • Fetch ONE node per request.
  • Group nodes into small chunks of CHUNK_SIZE (3) and put a CHUNK_PAUSE (2 s)
    gap between chunks so the IP never fires more than 3 req/2 s.
  • Retry each individual request up to MAX_RETRIES times with exponential
    backoff + jitter on 429 / 5xx / timeout.
  • On total failure for a node, skip it; the UI shows a partial-load warning.

This keeps the peak outbound request rate well below Open-Meteo's free limits
even from a shared Streamlit Cloud IP.

Example request (one node, Berlin):
    GET https://api.open-meteo.com/v1/forecast
        ?latitude=52.52&longitude=13.40
        &hourly=temperature_2m,windspeed_10m,windspeed_100m,
                winddirection_10m,shortwave_radiation,cloudcover,
                precipitation,surface_pressure
        &forecast_days=4&timezone=UTC
"""

import time
import logging
import random
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

# ─── Throttle / retry config ─────────────────────────────────────────────────
CHUNK_SIZE   = 3      # nodes per chunk
CHUNK_PAUSE  = 2.0    # seconds between chunks
NODE_PAUSE   = 0.4    # seconds between individual nodes within a chunk
MAX_RETRIES  = 3      # attempts per node (1 initial + 2 retries)
BACKOFF_BASE = 3.0    # seconds; doubled each retry
BACKOFF_MAX  = 20.0   # cap
RETRY_STATUS = {429, 500, 502, 503, 504}


# ══════════════════════════════════════════════════════════════════════════════
# SINGLE-NODE FETCH WITH RETRY
# ══════════════════════════════════════════════════════════════════════════════

def _fetch_one(node_id: str, meta: dict, client: httpx.Client) -> Optional[pd.DataFrame]:
    """
    Fetch a single node with up to MAX_RETRIES attempts.
    Returns a time-indexed DataFrame with metadata columns, or None on failure.
    """
    params = {
        "latitude":      meta["lat"],
        "longitude":     meta["lon"],
        "hourly":        ",".join(OPEN_METEO_VARIABLES),
        "forecast_days": (FORECAST_HOURS // 24) + 1,
        "timezone":      "UTC",
    }

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = client.get(OPEN_METEO_URL, params=params, timeout=15)

            # Retryable HTTP error
            if resp.status_code in RETRY_STATUS:
                if attempt < MAX_RETRIES:
                    wait = min(BACKOFF_BASE * attempt + random.uniform(0, 1), BACKOFF_MAX)
                    logger.warning("HTTP %s for %s (attempt %d/%d) – retry in %.1fs",
                                   resp.status_code, node_id, attempt, MAX_RETRIES, wait)
                    time.sleep(wait)
                    continue
                else:
                    logger.error("HTTP %s for %s – giving up", resp.status_code, node_id)
                    return None

            resp.raise_for_status()
            raw    = resp.json()
            hourly = raw.get("hourly", {})

            if not hourly or "time" not in hourly:
                logger.warning("Empty payload for %s", node_id)
                return None

            df = pd.DataFrame(hourly)
            df["time"] = pd.to_datetime(df["time"], utc=True)
            df = df.set_index("time").sort_index()

            now_utc = pd.Timestamp.now(tz="UTC").floor("h")
            df = df[df.index >= now_utc].head(FORECAST_HOURS)

            if df.empty:
                logger.warning("No future rows for %s", node_id)
                return None

            df["node_id"]   = node_id
            df["country"]   = meta["country"]
            df["subregion"] = meta["subregion"]
            df["label"]     = meta["label"]
            return df

        except httpx.TimeoutException:
            if attempt < MAX_RETRIES:
                wait = min(BACKOFF_BASE * attempt + random.uniform(0, 1), BACKOFF_MAX)
                logger.warning("Timeout for %s (attempt %d/%d) – retry in %.1fs",
                               node_id, attempt, MAX_RETRIES, wait)
                time.sleep(wait)
            else:
                logger.error("Timeout for %s – giving up", node_id)
                return None

        except Exception as exc:
            logger.error("Error fetching %s: %s", node_id, exc)
            return None

    return None   # exhausted retries


# ══════════════════════════════════════════════════════════════════════════════
# BULK FETCH – cached
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def fetch_all_nodes() -> Dict[str, pd.DataFrame]:
    """
    Fetch all configured nodes and return {node_id → DataFrame}.
    Cached for CACHE_TTL_SECONDS (1 h) so page refreshes are instant.

    Nodes are fetched in chunks of CHUNK_SIZE with CHUNK_PAUSE between chunks
    to stay within Open-Meteo's free-tier rate limits on shared IPs.
    """
    all_ids = list(NODES.keys())
    results: Dict[str, pd.DataFrame] = {}
    t0 = time.time()

    with httpx.Client() as client:
        for chunk_start in range(0, len(all_ids), CHUNK_SIZE):
            chunk = all_ids[chunk_start : chunk_start + CHUNK_SIZE]

            for node_id in chunk:
                df = _fetch_one(node_id, NODES[node_id], client)
                if df is not None:
                    results[node_id] = df
                time.sleep(NODE_PAUSE)

            # Pause between chunks (not after the very last one)
            if chunk_start + CHUNK_SIZE < len(all_ids):
                time.sleep(CHUNK_PAUSE)

    elapsed = time.time() - t0
    logger.info("Fetched %d/%d nodes in %.1fs", len(results), len(NODES), elapsed)
    return results


# ══════════════════════════════════════════════════════════════════════════════
# UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def get_fetch_timestamp() -> str:
    """Human-readable UTC timestamp for the dashboard footer."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def combined_dataframe(node_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Stack all per-node DataFrames into one long-form DataFrame."""
    if not node_data:
        return pd.DataFrame()
    return pd.concat(list(node_data.values())).sort_index()

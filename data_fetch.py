"""
data_fetch.py – Fetch hourly forecasts from Open-Meteo for all nodes.

KEY DESIGN DECISIONS vs. naive per-node loop
─────────────────────────────────────────────
1. BATCH API  – Open-Meteo accepts comma-separated latitude/longitude lists in
   a single request, so all 22 nodes are fetched in ONE HTTP call instead of 22.
   This is the primary fix for 429/502 errors on Streamlit Cloud's shared IPs:
   22 rapid-fire requests from the same IP trip rate limits; 1 request does not.

2. RETRY WITH BACKOFF – if the single batch call still fails (transient 502/503)
   we retry up to MAX_RETRIES times with exponential back-off + jitter before
   giving up and returning whatever partial data we already have.

3. GRACEFUL DEGRADATION – every caller receives a plain dict {node_id → DataFrame}.
   If the dict is empty (total API failure), app.py shows a user-friendly error
   banner instead of crashing with KeyError.

Example batch URL (abbreviated):
    GET https://api.open-meteo.com/v1/forecast
        ?latitude=51.46,52.52,48.14,...   (22 values)
        &longitude=7.01,13.40,11.58,...   (22 values)
        &hourly=temperature_2m,windspeed_10m,...
        &forecast_days=4
        &timezone=UTC
        &models=best_match              (optional – picks best model per location)

Open-Meteo returns a JSON *list* when multiple locations are requested.
"""

import time
import logging
import random
from datetime import datetime, timezone
from typing import Dict, List, Optional

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

# ─── Retry / back-off config ─────────────────────────────────────────────────
MAX_RETRIES    = 4          # total attempts (1 initial + 3 retries)
BACKOFF_BASE   = 2.0        # seconds; doubled each retry
BACKOFF_MAX    = 30.0       # cap
RETRY_STATUSES = {429, 500, 502, 503, 504}

# ─── Chunk size: how many nodes per batch request ────────────────────────────
# Open-Meteo supports up to ~50 locations per request; 22 fits in one call.
BATCH_SIZE = 22


# ══════════════════════════════════════════════════════════════════════════════
# INTERNAL HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _build_batch_params(node_ids: List[str]) -> dict:
    """Build the query-parameter dict for a batch of node_ids."""
    lats = ",".join(str(NODES[nid]["lat"]) for nid in node_ids)
    lons = ",".join(str(NODES[nid]["lon"]) for nid in node_ids)
    return {
        "latitude":      lats,
        "longitude":     lons,
        "hourly":        ",".join(OPEN_METEO_VARIABLES),
        "forecast_days": (FORECAST_HOURS // 24) + 1,
        "timezone":      "UTC",
    }


def _parse_location_response(raw: dict, node_id: str) -> Optional[pd.DataFrame]:
    """
    Parse a single location block from the Open-Meteo response into a
    time-indexed DataFrame, trim to FORECAST_HOURS, attach metadata.
    Returns None if the block is malformed or empty.
    """
    hourly = raw.get("hourly", {})
    if not hourly or "time" not in hourly:
        logger.warning("Malformed hourly block for %s", node_id)
        return None

    df = pd.DataFrame(hourly)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.set_index("time").sort_index()

    now_utc = pd.Timestamp.now(tz="UTC").floor("h")
    df = df[df.index >= now_utc].head(FORECAST_HOURS)

    if df.empty:
        logger.warning("No future rows for %s after trimming", node_id)
        return None

    meta = NODES[node_id]
    df["node_id"]   = node_id
    df["country"]   = meta["country"]
    df["subregion"] = meta["subregion"]
    df["label"]     = meta["label"]
    return df


def _fetch_batch(node_ids: List[str], client: httpx.Client) -> Dict[str, pd.DataFrame]:
    """
    Fetch one batch of nodes in a single HTTP request with retry/backoff.
    Returns a (possibly partial) dict of successful node DataFrames.
    """
    params = _build_batch_params(node_ids)
    results: Dict[str, pd.DataFrame] = {}

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = client.get(OPEN_METEO_URL, params=params, timeout=30)

            # On retryable HTTP errors, back off and try again
            if resp.status_code in RETRY_STATUSES and attempt < MAX_RETRIES:
                wait = min(BACKOFF_BASE ** attempt + random.uniform(0, 1), BACKOFF_MAX)
                logger.warning(
                    "HTTP %s on attempt %d/%d – retrying in %.1fs",
                    resp.status_code, attempt, MAX_RETRIES, wait,
                )
                time.sleep(wait)
                continue

            resp.raise_for_status()
            payload = resp.json()

            # Open-Meteo returns a list when multiple locations are requested,
            # or a single dict when only one location is requested.
            if isinstance(payload, dict):
                payload = [payload]

            for i, block in enumerate(payload):
                if i >= len(node_ids):
                    break
                nid = node_ids[i]
                df = _parse_location_response(block, nid)
                if df is not None:
                    results[nid] = df

            # Success – break out of retry loop
            break

        except httpx.TimeoutException:
            wait = min(BACKOFF_BASE ** attempt + random.uniform(0, 1), BACKOFF_MAX)
            logger.warning("Timeout on attempt %d/%d – retrying in %.1fs",
                           attempt, MAX_RETRIES, wait)
            if attempt < MAX_RETRIES:
                time.sleep(wait)

        except httpx.HTTPStatusError as exc:
            logger.error("HTTP %s – aborting batch: %s", exc.response.status_code, exc)
            break   # non-retryable or exhausted retries

        except Exception as exc:
            logger.error("Unexpected error fetching batch: %s", exc)
            break

    return results


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def fetch_all_nodes() -> Dict[str, pd.DataFrame]:
    """
    Fetch all configured nodes from Open-Meteo using batch requests and
    return a dict mapping node_id → time-indexed DataFrame.

    Strategy
    --------
    • Split all nodes into chunks of BATCH_SIZE (22 nodes = 1 request).
    • Each chunk is fetched in one HTTP call with retry/back-off on 429/5xx.
    • Results are cached for CACHE_TTL_SECONDS (1 h) so page refreshes are instant.
    • If *all* requests fail, returns an empty dict; callers must handle this
      gracefully (app.py shows an error banner).

    Example request built internally:
        GET https://api.open-meteo.com/v1/forecast
            ?latitude=51.46,52.52,48.14,48.85,45.75,43.30,...
            &longitude=7.01,13.40,11.58,2.35,4.84,5.37,...
            &hourly=temperature_2m,windspeed_10m,windspeed_100m,
                    winddirection_10m,shortwave_radiation,cloudcover,
                    precipitation,surface_pressure
            &forecast_days=4
            &timezone=UTC
    """
    all_node_ids = list(NODES.keys())
    results: Dict[str, pd.DataFrame] = {}
    t0 = time.time()

    with httpx.Client() as client:
        for chunk_start in range(0, len(all_node_ids), BATCH_SIZE):
            chunk = all_node_ids[chunk_start : chunk_start + BATCH_SIZE]
            batch_results = _fetch_batch(chunk, client)
            results.update(batch_results)
            # Small pause between chunks if there are multiple (future-proofing)
            if chunk_start + BATCH_SIZE < len(all_node_ids):
                time.sleep(1.0)

    elapsed = time.time() - t0
    logger.info("Fetched %d/%d nodes in %.1fs", len(results), len(NODES), elapsed)
    return results


def get_fetch_timestamp() -> str:
    """Human-readable UTC timestamp for the dashboard footer."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def combined_dataframe(node_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Stack all per-node DataFrames into one long-form DataFrame."""
    if not node_data:
        return pd.DataFrame()
    return pd.concat(list(node_data.values())).sort_index()
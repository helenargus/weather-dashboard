"""
config.py – European Power Market Weather Dashboard
All static configuration: nodes, countries, thresholds, model params.
"""

# ─── Regional Nodes ────────────────────────────────────────────────────────────
NODES = {
    # Germany – split into North/South sub-regions
    "DE-Essen":   {"lat": 51.46, "lon": 7.01,   "country": "DE", "subregion": "North", "label": "Essen"},
    "DE-Berlin":  {"lat": 52.52, "lon": 13.40,  "country": "DE", "subregion": "North", "label": "Berlin"},
    "DE-Munich":  {"lat": 48.14, "lon": 11.58,  "country": "DE", "subregion": "South", "label": "Munich"},
    # France
    "FR-Paris":     {"lat": 48.85, "lon": 2.35,  "country": "FR", "subregion": "North", "label": "Paris"},
    "FR-Lyon":      {"lat": 45.75, "lon": 4.84,  "country": "FR", "subregion": "Central", "label": "Lyon"},
    "FR-Marseille": {"lat": 43.30, "lon": 5.37,  "country": "FR", "subregion": "South", "label": "Marseille"},
    # UK
    "UK-London":     {"lat": 51.51, "lon": -0.13, "country": "UK", "subregion": "South", "label": "London"},
    "UK-Birmingham": {"lat": 52.48, "lon": -1.90, "country": "UK", "subregion": "Central", "label": "Birmingham"},
    "UK-Glasgow":    {"lat": 55.86, "lon": -4.25, "country": "UK", "subregion": "North",  "label": "Glasgow"},
    # Nordics
    "NO-Oslo":      {"lat": 59.91, "lon": 10.75, "country": "NO", "subregion": "South", "label": "Oslo"},
    "SE-Stockholm": {"lat": 59.33, "lon": 18.07, "country": "SE", "subregion": "Central","label": "Stockholm"},
    "FI-Helsinki":  {"lat": 60.17, "lon": 24.94, "country": "FI", "subregion": "South", "label": "Helsinki"},
    # Iberia
    "ES-Madrid":    {"lat": 40.42, "lon": -3.70, "country": "ES", "subregion": "Central","label": "Madrid"},
    "ES-Barcelona": {"lat": 41.39, "lon": 2.17,  "country": "ES", "subregion": "East",   "label": "Barcelona"},
    "PT-Lisbon":    {"lat": 38.72, "lon": -9.14, "country": "PT", "subregion": "West",   "label": "Lisbon"},
    # Italy
    "IT-Milan":  {"lat": 45.46, "lon": 9.19,  "country": "IT", "subregion": "North", "label": "Milan"},
    "IT-Rome":   {"lat": 41.90, "lon": 12.50, "country": "IT", "subregion": "Central","label": "Rome"},
    "IT-Naples": {"lat": 40.85, "lon": 14.27, "country": "IT", "subregion": "South", "label": "Naples"},
    # CEE
    "PL-Warsaw":  {"lat": 52.23, "lon": 21.01, "country": "PL", "subregion": "Central","label": "Warsaw"},
    "CZ-Prague":  {"lat": 50.08, "lon": 14.43, "country": "CZ", "subregion": "West",   "label": "Prague"},
    "AT-Vienna":  {"lat": 48.21, "lon": 16.37, "country": "AT", "subregion": "East",   "label": "Vienna"},
    "HU-Budapest":{"lat": 47.50, "lon": 19.04, "country": "HU", "subregion": "Central","label": "Budapest"},
}

COUNTRY_LABELS = {
    "DE": "Germany 🇩🇪",
    "FR": "France 🇫🇷",
    "UK": "UK 🇬🇧",
    "NO": "Norway 🇳🇴",
    "SE": "Sweden 🇸🇪",
    "FI": "Finland 🇫🇮",
    "ES": "Spain 🇪🇸",
    "PT": "Portugal 🇵🇹",
    "IT": "Italy 🇮🇹",
    "PL": "Poland 🇵🇱",
    "CZ": "Czechia 🇨🇿",
    "AT": "Austria 🇦🇹",
    "HU": "Hungary 🇭🇺",
}

# Group countries for spread analysis
COUNTRY_GROUPS = {
    "Continental Core": ["DE", "FR", "AT", "CZ"],
    "Nordics":          ["NO", "SE", "FI"],
    "Iberia":           ["ES", "PT"],
    "British Isles":    ["UK"],
    "Italy":            ["IT"],
    "CEE":              ["PL", "CZ", "AT", "HU"],
}

# ─── Forecast windows ──────────────────────────────────────────────────────────
FORECAST_HOURS = 72          # 0-72h horizon
CACHE_TTL_SECONDS = 3600     # 1-hour cache

# ─── Open-Meteo API ────────────────────────────────────────────────────────────
OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_VARIABLES = [
    "temperature_2m",
    "windspeed_10m",
    "windspeed_100m",
    "winddirection_10m",
    "shortwave_radiation",
    "cloudcover",
    "precipitation",
    "surface_pressure",
]

# ─── Power Curve (simple piece-wise linear, MW per node) ──────────────────────
# Represents a generic onshore wind farm capacity factor approximation
WIND_CUT_IN   = 3.0   # m/s
WIND_RATED    = 12.0  # m/s  → 100% CF
WIND_CUT_OUT  = 25.0  # m/s

# ─── Temperature thresholds (°C) ──────────────────────────────────────────────
BALANCE_TEMP   = 15.0   # base temp for HDD/CDD calculation
COLD_THRESHOLD = 5.0    # below → strong heating demand signal
HEAT_THRESHOLD = 25.0   # above → strong cooling demand signal

# ─── Signal scoring weights ────────────────────────────────────────────────────
WEIGHTS = {
    "wind":   0.35,
    "solar":  0.30,
    "demand": 0.35,
}

# Threshold for bullish/bearish label
SCORE_BULL_THRESHOLD =  0.20   # net positive supply or demand signal
SCORE_BEAR_THRESHOLD = -0.20

# ─── Ramp detection ───────────────────────────────────────────────────────────
WIND_RAMP_THRESHOLD_MW  = 0.25   # CF change ≥ 25pp in 3h → flag
SOLAR_RAMP_THRESHOLD_MW = 0.30
TEMP_RAMP_THRESHOLD     = 4.0    # °C change in 3h

# ─── Spread pairs for inter-regional analysis ─────────────────────────────────
SPREAD_PAIRS = [
    ("DE", "FR",  "DE–FR Spread"),
    ("DE", "UK",  "Continent–UK"),
    ("NO", "DE",  "Hydro Export Signal (NO→DE)"),
    ("ES", "FR",  "Iberia–France"),
    ("IT", "AT",  "Italy–CEE"),
    ("PL", "DE",  "CEE–Germany"),
]

# ─── UI ───────────────────────────────────────────────────────────────────────
APP_TITLE   = "⚡ EU Power Weather Dashboard"
APP_ICON    = "⚡"
THEME_COLOR = "#00D4FF"        # accent cyan
BEAR_COLOR  = "#FF4B4B"
BULL_COLOR  = "#00C853"
NEUTRAL_COLOR = "#FFA726"

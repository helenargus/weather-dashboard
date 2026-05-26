# ⚡ EU Power Market Weather Dashboard

A production-ready European power market weather intelligence dashboard built with Python and Streamlit. Converts live Open-Meteo weather forecasts into actionable power-trading signals across 22 nodes in 13 countries — **completely free, no API keys, no paid services**.

---

## Features

| Feature | Detail |
|---|---|
| **Coverage** | 22 nodes: DE, FR, UK, NO, SE, FI, ES, PT, IT, PL, CZ, AT, HU |
| **Wind proxy** | Piece-wise linear power curve → capacity factor 0–1 |
| **Solar proxy** | Shortwave radiation × cloud-cover penalty → CF 0–1 |
| **Demand model** | HDD / CDD from 15°C balance temp → demand pressure signal |
| **Composite score** | Weighted net bull/bear signal per node, country, sub-region |
| **Spread signals** | DE–FR, UK–Continent, NO–DE, Iberia–France, CEE–Germany |
| **Intra-country** | North vs South Germany, etc. |
| **Ramp detection** | ≥25pp wind CF change in 3h, ≥30pp solar, ≥4°C temp |
| **Trade Insight Panel** | Auto-generated bullet insights at top of dashboard |
| **0–72h charts** | Wind, solar, temp timeseries with ramp markers |
| **Caching** | `st.cache_data` TTL = 1h → fast page loads |
| **Dark UI** | Trader-focused dark mode, Plotly dark charts |

---

## Run Locally (2 minutes)

```bash
# 1. Clone or download this repository
git clone https://github.com/YOUR_USERNAME/eu-power-dashboard.git
cd eu-power-dashboard

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch the app
streamlit run app.py
```

The dashboard will open at **http://localhost:8501**.

---

## Deploy for Free on Streamlit Community Cloud (Permanent Public URL)

Follow these steps to get a live public URL at `https://share.streamlit.io` — **100% free, runs 24/7**.

### Step 1 — Create a GitHub Account
Go to https://github.com and sign up (free).

### Step 2 — Create a New GitHub Repository

1. Click **"New repository"** (green button) on github.com.
2. Name it something like `eu-power-dashboard`.
3. Set it to **Public** (required for free Streamlit Cloud).
4. Click **"Create repository"**.

### Step 3 — Upload Your Files to GitHub

**Option A: GitHub Web UI (easiest)**

1. In your new repository, click **"Add file" → "Upload files"**.
2. Drag and drop all 6 files:
   - `app.py`
   - `config.py`
   - `data_fetch.py`
   - `transformations.py`
   - `requirements.txt`
   - `README.md`
3. Click **"Commit changes"**.

**Option B: Git command line**

```bash
cd eu-power-dashboard
git init
git add .
git commit -m "Initial commit: EU power dashboard"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/eu-power-dashboard.git
git push -u origin main
```

### Step 4 — Deploy on Streamlit Community Cloud

1. Go to **https://share.streamlit.io** and sign in with GitHub.
2. Click **"New app"**.
3. Fill in the form:
   - **Repository**: `YOUR_USERNAME/eu-power-dashboard`
   - **Branch**: `main`
   - **Main file path**: `app.py`
4. Click **"Deploy!"**

Streamlit Cloud will:
- Install dependencies from `requirements.txt` automatically
- Launch the app (takes ~2 minutes on first deploy)
- Give you a permanent public URL like:
  `https://your-username-eu-power-dashboard-app-xxxx.streamlit.app`

### Step 5 — Share the URL

Copy the URL and share it. The app runs 24/7 for free. It will auto-restart if it goes to sleep after inactivity.

---

## Project Structure

```
eu-power-dashboard/
├── app.py              ← Streamlit UI entry point (run this)
├── config.py           ← All constants: nodes, thresholds, weights
├── data_fetch.py       ← Open-Meteo API calls with st.cache_data
├── transformations.py  ← Wind/solar/demand models, scoring, insights
├── requirements.txt    ← Python dependencies
└── README.md           ← This file
```

---

## API Details

This dashboard uses the **Open-Meteo API** — free, no registration, no API key.

### Example API call (one node):

```
GET https://api.open-meteo.com/v1/forecast
    ?latitude=52.52
    &longitude=13.40
    &hourly=temperature_2m,windspeed_10m,windspeed_100m,
             winddirection_10m,shortwave_radiation,cloudcover,
             precipitation,surface_pressure
    &forecast_days=4
    &timezone=UTC
```

The app fetches all 22 nodes on startup and caches results for 1 hour.

---

## Signal Methodology

### Wind Power Proxy
```
CF = 0                           if wind < 3 m/s (cut-in)
CF = (wind - 3) / (12 - 3)      if 3 ≤ wind ≤ 12 m/s (ramp)
CF = 1                           if 12 < wind ≤ 25 m/s (rated)
CF = 0                           if wind > 25 m/s (cut-out)
```

### Solar Proxy
```
CF = (radiation / 900 W/m²) × (1 - cloudcover% × 0.75)
```

### Demand Signal
```
HDD = max(0, 15 - T)
CDD = max(0, T - 15)
demand_signal = (HDD - CDD) / 10   clipped to [-1, +1]
```

### Composite Score (per node)
```
score = -0.35 × wind_CF - 0.30 × solar_CF + 0.35 × demand_signal
```
Positive score = bullish price pressure (supply short or demand high).  
Negative score = bearish price pressure (supply long or demand low).

### Thresholds
| Signal | Range |
|---|---|
| 🟢 BULLISH | score ≥ +0.20 |
| 🟡 NEUTRAL | -0.20 < score < +0.20 |
| 🔴 BEARISH | score ≤ -0.20 |

---

## Updating the App

To update after deployment, simply push changes to GitHub:

```bash
git add .
git commit -m "Update: description of change"
git push
```

Streamlit Cloud auto-deploys within ~30 seconds.

---

## Disclaimer

This dashboard is for **informational and educational purposes only**. It does not constitute financial or trading advice. Weather-to-power models are simplified proxies; always validate with official grid and market data before trading.

---

## Data Source

[Open-Meteo](https://open-meteo.com) — Free weather API for non-commercial use.  
License: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

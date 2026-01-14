# Outbreaks

Tagline: Preempting Waterborne Disease Outbreaks with AI and Satellite Data.

## Core Problem
Over 2 billion people use a drinking water source contaminated with feces. Waterborne diseases like cholera, typhoid, and dysentery cause hundreds of thousands of deaths annually, often in low-resource communities. Outbreaks are traditionally detected reactively--after people get sick--leading to delayed and costly responses.

## Technical Solution
Outbreaks is an early-warning predictive modeling platform that shifts response from reactive to proactive.

### Data Fusion Engine
Aggregates and processes multi-modal data:

- Remote Sensing: NASA/USGS/ESA satellite data on sea surface temperature, chlorophyll-a levels (algae bloom proxy), precipitation, and flood inundation maps.
- Climate and Weather: NOAA forecasts, historical precipitation patterns, and drought indices.
- Socio-economic and Infrastructure Data: Population density, access to improved water sources (JMP data), and sanitation infrastructure from local governments and NGOs.
- Crowdsourced and Sentinel Data: Anonymous and aggregated mobility data to track population movement post-flooding, and local water quality reports from partner health clinics.

### Predictive AI Model
An ensemble model (e.g., Gradient Boosting plus LSTMs) trained on historical outbreak data. It identifies high-risk "hotspots" by correlating environmental precursors (e.g., a heatwave followed by heavy flooding) with a high probability of pathogen contamination in water supplies.

### Actionable Dashboard and Alerts
Provides a clear, GIS-based interface for public health officials at NGOs and government agencies. The system sends targeted SMS alerts to community health workers in predicted high-risk areas weeks before a potential outbreak.

## Impact and Equity Focus

- Saves Lives: Drastically reduces morbidity and mortality from preventable diseases.
- Resource Optimization: Allows NGOs and governments to pre-position water purification tablets, medical supplies, and health teams before a crisis hits, maximizing the impact of limited aid dollars.
- Closes the Data Gap: Brings advanced, space-age analytics to the most vulnerable communities who lack on-the-ground water testing resources.

## MVP: Predictive Risk Scoring
This proof-of-concept trains a model on synthetic, multi-modal features and predicts a Waterborne Disease Risk Score (0-100) for a given latitude, longitude, and date.

### How it Works
1. Generate synthetic data that mimics environmental + socio-economic drivers.
2. Train a tree-based regressor (XGBoost if available, else GradientBoosting).
3. Score new coordinates and render a Folium risk map.

### Quick Start
```bash
pip install -r requirements.txt
python run.py
```

### Web App
```bash
python webapp.py
```
Open `http://localhost:8001` in your browser.

### API (REST)
Endpoints (base URL `http://localhost:8001/api`):
- `POST /score` with JSON `{ "lat": 0.5, "lon": 32.5, "date": "2024-01-10" }`
- `POST /score/batch` with JSON list or `file=@points.csv`
- `GET /points?limit=1500&start=2023-01-01&end=2023-12-31`
- `GET /export/csv`
- `GET /export/pdf`

### Docker (One Command)
```bash
docker compose up --build
```
Then open `http://localhost:8001`.

### GitHub Pages (Static Demo)
The `docs/` folder contains a static demo suitable for GitHub Pages.
It uses precomputed data in `docs/data/` and does not run the live API.

Enable Pages in GitHub:
- Source: `main` branch
- Folder: `/docs`

Optional NASA POWER sample (requires network access):
```bash
USE_NASA_POWER=true python run.py
```

CLI overrides:
```bash
python run.py --use-nasa-power --power-grid-size 6 --power-bbox "-5,5,28,38"
```
Enable mock GEE overlays:
```bash
python run.py --use-gee-mock
```

Outputs are written to `results/`:
- `risk_scored_points.csv`
- `risk_map.html`
- `nasa_power_sample.csv` (when `USE_NASA_POWER=true`)
- `nasa_power_cache.csv` (cached NASA POWER response)
- `model_report.json`
- `model_diagnostics_fit.png`
- `model_diagnostics_residuals.png`

### Research-Grade Extras
- Cross-validated model selection across linear, gradient boosting, random forest, and XGBoost (if installed).
- Holdout evaluation metrics (MAE, RMSE, R2) plus top feature importance report.
- Diagnostics: fit scatter and residual distribution plots.
- Calibration layer (linear) on top of the selected model.
- Prediction intervals from quantile regression models.
- Time-series features (rolling precipitation, SST, flood metrics) per spatial bin.

### Real-Data Connectors (Stubs)
- NASA POWER API URL builder + parser in `data/nasa_power.py` (network fetch gated by `allow_network=True`).
- Google Earth Engine interface placeholders in `data/gee_interface.py`.
- Mock GEE raster sampling in `data/gee_mock.py` (enabled by default for chlorophyll + flood overlays).

When `USE_NASA_POWER=true`, daily precipitation and air temperature are merged into the synthetic training set using a small latitude/longitude grid and inverse-distance weighting.

POWER configuration lives in `config/settings.py`:
- `POWER_BBOX` (grid coverage)
- `POWER_GRID_SIZE` (points per axis)
- `POWER_START_DATE` / `POWER_END_DATE`
- `POWER_USE_DATASET_BBOX` (align the POWER grid to the synthetic dataset bbox)

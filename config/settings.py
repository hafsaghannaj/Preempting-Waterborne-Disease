import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
MODEL_PATH = os.path.join(RESULTS_DIR, "risk_model.joblib")
RANDOM_SEED = 42

DEFAULT_BBOX = {
    "lat_min": -10.0,
    "lat_max": 10.0,
    "lon_min": 20.0,
    "lon_max": 50.0,
}

POWER_BBOX = {
    "lat_min": -5.0,
    "lat_max": 5.0,
    "lon_min": 28.0,
    "lon_max": 38.0,
}
POWER_USE_DATASET_BBOX = True
POWER_GRID_SIZE = 4
POWER_START_DATE = "2022-01-01"
POWER_END_DATE = "2022-01-31"

GEE_MOCK_ENABLED = True

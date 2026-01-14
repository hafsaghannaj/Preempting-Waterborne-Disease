import hashlib
from datetime import datetime

import numpy as np


def _stable_seed(lat, lon, date_str):
    seed_str = f"{lat:.4f}:{lon:.4f}:{date_str}"
    digest = hashlib.sha256(seed_str.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def simulate_features(lat, lon, date):
    if isinstance(date, str):
        date_obj = datetime.strptime(date, "%Y-%m-%d")
    else:
        date_obj = date

    seed = _stable_seed(lat, lon, date_obj.strftime("%Y-%m-%d"))
    rng = np.random.RandomState(seed)

    seasonal = np.sin((date_obj.timetuple().tm_yday / 365.0) * 2 * np.pi)
    sst = 24 + (lat / 10.0) + 2.5 * seasonal + rng.normal(0, 0.7)
    chlor_a = np.clip(0.6 + 0.2 * seasonal + rng.normal(0, 0.1), 0.05, 2.0)
    precip = np.clip(80 + 60 * seasonal + rng.normal(0, 25), 0, 250)
    flood_inundation = np.clip((precip - 120) / 130 + rng.normal(0, 0.1), 0, 1)
    drought_index = np.clip(-1.0 * seasonal + rng.normal(0, 0.4), -2.5, 2.5)

    pop_density = np.clip(300 + (abs(lat) * 40) + rng.normal(0, 50), 50, 1200)
    water_access_pct = np.clip(70 - (abs(lat) * 1.5) + rng.normal(0, 4), 40, 98)
    sanitation_score = np.clip(65 - (abs(lat) * 1.2) + rng.normal(0, 6), 30, 95)

    mobility_index = np.clip(1 + flood_inundation * 1.5 + rng.normal(0, 0.2), 0.5, 3.0)
    clinic_reports = np.clip(0.5 + flood_inundation * 2.2 + rng.normal(0, 0.4), 0, 6)

    return {
        "sst": sst,
        "chlor_a": chlor_a,
        "precip": precip,
        "flood_inundation": flood_inundation,
        "drought_index": drought_index,
        "population_density": pop_density,
        "water_access_pct": water_access_pct,
        "sanitation_score": sanitation_score,
        "mobility_index": mobility_index,
        "clinic_reports": clinic_reports,
    }

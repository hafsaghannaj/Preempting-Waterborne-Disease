from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from config.settings import DEFAULT_BBOX, RANDOM_SEED
from data.gee_mock import mock_chlorophyll, mock_flood_extent
from utils.data_simulator import simulate_features


def _random_date(rng, start_date, end_date):
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    delta_days = (end - start).days
    return start + timedelta(days=int(rng.randint(0, max(delta_days, 1))))


def _risk_score(features, rng):
    heatwave = 1 if features["sst"] > 28 else 0
    flood = features["flood_inundation"]
    low_water_access = max(0, (75 - features["water_access_pct"]) / 30)
    sanitation_risk = max(0, (70 - features["sanitation_score"]) / 40)
    mobility = features["mobility_index"]
    clinic = features["clinic_reports"]

    raw = (
        0.25 * heatwave
        + 0.3 * flood
        + 0.2 * low_water_access
        + 0.15 * sanitation_risk
        + 0.1 * mobility
        + 0.1 * clinic
    )
    noisy = raw + rng.normal(0, 0.05)
    return float(np.clip(noisy * 100, 0, 100))


def _apply_power_overlay(lat, lon, date_obj, features, power_lookup):
    if not power_lookup:
        return

    date_key = date_obj.date()
    frame = power_lookup.get(date_key)
    if frame is None or frame.empty:
        return

    if "lat" in frame and "lon" in frame:
        points = frame[["lat", "lon"]].to_numpy()
        distances = np.sqrt((points[:, 0] - lat) ** 2 + (points[:, 1] - lon) ** 2)
        weights = 1.0 / (distances + 1e-3)
        for source, target in (("precip", "precip"), ("air_temp", "sst")):
            if source in frame:
                values = frame[source].to_numpy()
                features[target] = float(np.average(values, weights=weights))
    else:
        row = frame.iloc[0]
        if "precip" in row:
            features["precip"] = float(row["precip"])
        if "air_temp" in row:
            features["sst"] = float(row["air_temp"])


def generate_synthetic_dataset(
    n_samples=1000,
    start_date="2021-01-01",
    end_date="2023-12-31",
    bbox=None,
    seed=RANDOM_SEED,
    power_df=None,
    use_gee_mock=False,
    gee_bbox=None,
    n_locations=60,
    samples_per_location=None,
):
    if bbox is None:
        bbox = DEFAULT_BBOX

    rng = np.random.RandomState(seed)
    rows = []

    power_lookup = None
    if power_df is not None and not power_df.empty:
        frame = power_df.copy()
        frame["date"] = pd.to_datetime(frame["date"]).dt.date
        power_lookup = {date: group for date, group in frame.groupby("date")}

    locations = [
        (
            rng.uniform(bbox["lat_min"], bbox["lat_max"]),
            rng.uniform(bbox["lon_min"], bbox["lon_max"]),
        )
        for _ in range(n_locations)
    ]

    if samples_per_location is not None:
        total_samples = n_locations * samples_per_location
    else:
        total_samples = n_samples

    for _ in range(total_samples):
        lat, lon = locations[int(rng.randint(0, len(locations)))]
        date = _random_date(rng, start_date, end_date)
        features = simulate_features(lat, lon, date)
        if use_gee_mock:
            features["chlor_a"] = mock_chlorophyll(lat, lon, date, bbox=gee_bbox)
            features["flood_inundation"] = mock_flood_extent(lat, lon, date, bbox=gee_bbox)
        _apply_power_overlay(lat, lon, date, features, power_lookup)
        risk_score = _risk_score(features, rng)

        row = {
            "lat": lat,
            "lon": lon,
            "date": date.strftime("%Y-%m-%d"),
            **features,
            "risk_score": risk_score,
        }
        rows.append(row)

    return pd.DataFrame(rows)

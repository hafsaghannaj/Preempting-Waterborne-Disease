import json
import os

import pandas as pd

from config.settings import GEE_MOCK_ENABLED, MODEL_PATH, RESULTS_DIR
from data.synthetic_data import generate_synthetic_dataset
from models.model_train import train_model
from predictor.aqua_predictor import AquaSentinelPredictor


def ensure_artifacts():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    if not os.path.exists(MODEL_PATH):
        data = generate_synthetic_dataset(n_samples=1400, use_gee_mock=GEE_MOCK_ENABLED)
        report_path = os.path.join(RESULTS_DIR, "model_report.json")
        train_model(data, MODEL_PATH, report_path=report_path)

    points_path = os.path.join(RESULTS_DIR, "risk_scored_points.csv")
    if not os.path.exists(points_path):
        data = generate_synthetic_dataset(
            n_samples=2400,
            n_locations=80,
            samples_per_location=20,
            use_gee_mock=GEE_MOCK_ENABLED,
        )
        predictor = AquaSentinelPredictor(MODEL_PATH)
        data["risk_score"], data["interval_lower"], data["interval_upper"] = zip(
            *data.apply(
                lambda row: predictor.predict_with_interval(
                    row["lat"], row["lon"], row["date"]
                ),
                axis=1,
            )
        )
        data.to_csv(points_path, index=False)

    report_path = os.path.join(RESULTS_DIR, "model_report.json")
    if not os.path.exists(report_path):
        with open(report_path, "w", encoding="utf-8") as handle:
            json.dump({}, handle)

    return points_path


def load_points(limit=None, start_date=None, end_date=None):
    points_path = ensure_artifacts()
    df = pd.read_csv(points_path)

    if start_date:
        df = df[df["date"] >= start_date]
    if end_date:
        df = df[df["date"] <= end_date]
    if limit:
        df = df.head(limit)

    return df

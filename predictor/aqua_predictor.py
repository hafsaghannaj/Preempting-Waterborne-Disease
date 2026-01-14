from datetime import datetime

import joblib
import pandas as pd

from utils.data_simulator import simulate_features
from utils.feature_engineer import build_feature_frame


class AquaSentinelPredictor:
    def __init__(self, model_path):
        payload = joblib.load(model_path)
        self.model = payload["model"]
        self.feature_cols = payload["feature_cols"]
        self.calibrator = payload.get("calibrator")
        self.lower_model = payload.get("lower_model")
        self.upper_model = payload.get("upper_model")

    def predict_risk(self, lat, lon, date):
        if isinstance(date, str):
            date_obj = datetime.strptime(date, "%Y-%m-%d")
        else:
            date_obj = date

        features = simulate_features(lat, lon, date_obj)
        record = {
            "lat": lat,
            "lon": lon,
            "date": date_obj.strftime("%Y-%m-%d"),
            **features,
        }
        df = pd.DataFrame([record])
        engineered, _ = build_feature_frame(df)
        X = engineered[self.feature_cols]
        score = float(self.model.predict(X)[0])
        if self.calibrator is not None:
            score = float(self.calibrator.predict([[score]])[0])
        return max(0.0, min(100.0, score))

    def predict_with_interval(self, lat, lon, date):
        if isinstance(date, str):
            date_obj = datetime.strptime(date, "%Y-%m-%d")
        else:
            date_obj = date

        features = simulate_features(lat, lon, date_obj)
        record = {
            "lat": lat,
            "lon": lon,
            "date": date_obj.strftime("%Y-%m-%d"),
            **features,
        }
        df = pd.DataFrame([record])
        engineered, _ = build_feature_frame(df)
        X = engineered[self.feature_cols]

        score = float(self.model.predict(X)[0])
        if self.calibrator is not None:
            score = float(self.calibrator.predict([[score]])[0])

        lower = None
        upper = None
        if self.lower_model is not None and self.upper_model is not None:
            lower = float(self.lower_model.predict(X)[0])
            upper = float(self.upper_model.predict(X)[0])

        return (
            max(0.0, min(100.0, score)),
            max(0.0, min(100.0, lower)) if lower is not None else None,
            max(0.0, min(100.0, upper)) if upper is not None else None,
        )

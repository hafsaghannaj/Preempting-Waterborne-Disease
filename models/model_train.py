import json
import joblib
import numpy as np
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.model_selection import KFold, train_test_split

from utils.feature_engineer import build_feature_frame


try:
    from xgboost import XGBRegressor

    _HAS_XGBOOST = True
except Exception:
    XGBRegressor = None
    _HAS_XGBOOST = False


def _candidate_models():
    models = {
        "linear_regression": LinearRegression(),
        "gradient_boosting": GradientBoostingRegressor(random_state=42),
        "random_forest": RandomForestRegressor(
            n_estimators=240, max_depth=12, random_state=42
        ),
    }
    if _HAS_XGBOOST:
        models["xgboost"] = XGBRegressor(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.06,
            subsample=0.8,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            random_state=42,
        )
    return models


def _cross_validate(models, X, y, folds=3):
    cv = KFold(n_splits=folds, shuffle=True, random_state=42)
    results = {}

    for name, model in models.items():
        fold_metrics = []
        for train_idx, val_idx in cv.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            fold_model = clone(model)
            fold_model.fit(X_train, y_train)
            preds = fold_model.predict(X_val)
            fold_metrics.append(
                {
                    "mae": mean_absolute_error(y_val, preds),
                    "rmse": float(np.sqrt(mean_squared_error(y_val, preds))),
                    "r2": r2_score(y_val, preds),
                }
            )
        results[name] = {
            metric: float(np.mean([m[metric] for m in fold_metrics]))
            for metric in fold_metrics[0]
        }

    return results


def _select_best(cv_results):
    return min(cv_results.items(), key=lambda item: item[1]["mae"])[0]


def _fit_calibration(y_true, preds):
    calibrator = LinearRegression()
    calibrator.fit(preds.reshape(-1, 1), y_true)
    return calibrator


def _apply_calibration(calibrator, preds):
    return calibrator.predict(preds.reshape(-1, 1))


def _fit_quantile_models(X, y):
    lower = GradientBoostingRegressor(
        loss="quantile", alpha=0.1, random_state=42
    ).fit(X, y)
    upper = GradientBoostingRegressor(
        loss="quantile", alpha=0.9, random_state=42
    ).fit(X, y)
    return lower, upper


def train_model(df, model_path, report_path=None):
    engineered, feature_cols = build_feature_frame(df)
    X = engineered[feature_cols]
    y = engineered["risk_score"]

    models = _candidate_models()
    cv_results = _cross_validate(models, X, y)
    best_name = _select_best(cv_results)

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    X_train, X_calib, y_train, y_calib = train_test_split(
        X_train_full, y_train_full, test_size=0.2, random_state=42
    )

    model = models[best_name]
    model.fit(X_train, y_train)

    calib_preds = model.predict(X_calib)
    calibrator = _fit_calibration(y_calib, calib_preds)

    preds = model.predict(X_test)
    calibrated_preds = _apply_calibration(calibrator, preds)

    metrics = {
        "selected_model": best_name,
        "r2": float(r2_score(y_test, calibrated_preds)),
        "mae": float(mean_absolute_error(y_test, calibrated_preds)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, calibrated_preds))),
        "cv_results": cv_results,
    }

    lower_model, upper_model = _fit_quantile_models(X_train_full, y_train_full)
    lower_preds = lower_model.predict(X_test)
    upper_preds = upper_model.predict(X_test)
    interval_coverage = float(
        np.mean((y_test >= lower_preds) & (y_test <= upper_preds))
    )
    interval_width = float(np.mean(upper_preds - lower_preds))
    metrics["interval_coverage"] = interval_coverage
    metrics["interval_width"] = interval_width

    feature_importance = None
    if hasattr(model, "feature_importances_"):
        feature_importance = dict(zip(feature_cols, model.feature_importances_))
    elif hasattr(model, "coef_"):
        coef = np.array(model.coef_).ravel()
        feature_importance = dict(zip(feature_cols, np.abs(coef)))

    payload = {
        "model": model,
        "calibrator": calibrator,
        "lower_model": lower_model,
        "upper_model": upper_model,
        "feature_cols": feature_cols,
    }
    joblib.dump(payload, model_path)

    if report_path:
        report = {
            "metrics": metrics,
            "top_features": None,
        }
        if feature_importance:
            ranked = sorted(feature_importance.items(), key=lambda item: item[1], reverse=True)
            report["top_features"] = [
                {"feature": name, "importance": float(value)} for name, value in ranked[:8]
            ]
        with open(report_path, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)

    diagnostics = {
        "y_test": y_test,
        "preds": calibrated_preds,
        "interval_lower": lower_preds,
        "interval_upper": upper_preds,
    }

    return metrics, diagnostics

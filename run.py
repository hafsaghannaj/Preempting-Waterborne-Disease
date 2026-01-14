import argparse
import os

from config.settings import (
    MODEL_PATH,
    RESULTS_DIR,
    DEFAULT_BBOX,
    POWER_BBOX,
    POWER_END_DATE,
    POWER_GRID_SIZE,
    POWER_START_DATE,
    POWER_USE_DATASET_BBOX,
    GEE_MOCK_ENABLED,
)
from data.nasa_power import load_or_fetch_power_grid
from data.synthetic_data import generate_synthetic_dataset
from models.model_train import train_model
from predictor.aqua_predictor import AquaSentinelPredictor
from visualization.model_diagnostics import save_diagnostic_plots
from visualization.risk_mapper import generate_risk_map


def _parse_bbox(value):
    parts = [p.strip() for p in value.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(
            "bbox must be four comma-separated values: lat_min,lat_max,lon_min,lon_max"
        )
    lat_min, lat_max, lon_min, lon_max = [float(p) for p in parts]
    return {
        "lat_min": lat_min,
        "lat_max": lat_max,
        "lon_min": lon_min,
        "lon_max": lon_max,
    }


def _parse_args():
    parser = argparse.ArgumentParser(description="AquaSentinel MVP runner")
    parser.add_argument(
        "--use-nasa-power",
        action="store_true",
        help="Fetch NASA POWER data and merge into synthetic features.",
    )
    parser.add_argument(
        "--power-grid-size",
        type=int,
        default=POWER_GRID_SIZE,
        help="Number of grid points per axis for NASA POWER.",
    )
    parser.add_argument(
        "--power-bbox",
        type=_parse_bbox,
        help="Override NASA POWER bbox: lat_min,lat_max,lon_min,lon_max",
    )
    parser.add_argument(
        "--use-gee-mock",
        action="store_true",
        help="Override chlorophyll/flood with mock GEE raster values.",
    )
    return parser.parse_args()


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    dataset_bbox = DEFAULT_BBOX
    args = _parse_args()
    env_use_nasa = os.getenv("USE_NASA_POWER", "false").lower() == "true"
    use_nasa = args.use_nasa_power or env_use_nasa
    use_gee_mock = args.use_gee_mock or GEE_MOCK_ENABLED
    power_df = None
    if use_nasa:
        try:
            cache_path = os.path.join(RESULTS_DIR, "nasa_power_cache.csv")
            power_bbox = args.power_bbox
            if power_bbox is None:
                power_bbox = dataset_bbox if POWER_USE_DATASET_BBOX else POWER_BBOX
            power_df = load_or_fetch_power_grid(
                bbox=power_bbox,
                start_date=POWER_START_DATE,
                end_date=POWER_END_DATE,
                grid_size=args.power_grid_size,
                cache_path=cache_path,
                allow_network=True,
                verbose=True,
            )
            power_path = os.path.join(RESULTS_DIR, "nasa_power_sample.csv")
            power_df.to_csv(power_path, index=False)
            print(f"Saved NASA POWER sample: {power_path}")
        except Exception as exc:
            print(f"NASA POWER fetch failed, continuing with synthetic data: {exc}")

    data = generate_synthetic_dataset(
        n_samples=1200,
        bbox=dataset_bbox,
        power_df=power_df,
        use_gee_mock=use_gee_mock,
        gee_bbox=dataset_bbox,
    )
    report_path = os.path.join(RESULTS_DIR, "model_report.json")
    metrics, diagnostics = train_model(data, MODEL_PATH, report_path=report_path)

    predictor = AquaSentinelPredictor(MODEL_PATH)
    sample = data.sample(150, random_state=24).copy()
    sample["risk_score"] = sample.apply(
        lambda row: predictor.predict_risk(row["lat"], row["lon"], row["date"]),
        axis=1,
    )

    csv_path = os.path.join(RESULTS_DIR, "risk_scored_points.csv")
    sample.to_csv(csv_path, index=False)

    map_path = os.path.join(RESULTS_DIR, "risk_map.html")
    generate_risk_map(sample, map_path)

    diagnostic_prefix = os.path.join(RESULTS_DIR, "model_diagnostics")
    save_diagnostic_plots(diagnostics["y_test"], diagnostics["preds"], diagnostic_prefix)

    print("Training metrics:")
    print(metrics)
    print(f"Saved model report: {report_path}")
    print(f"Saved scored points: {csv_path}")
    print(f"Saved risk map: {map_path}")


if __name__ == "__main__":
    main()

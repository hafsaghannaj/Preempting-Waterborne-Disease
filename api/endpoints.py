import io
import json
import os
from datetime import datetime

import pandas as pd
from flask import Blueprint, jsonify, request, send_file

from config.settings import MODEL_PATH, RESULTS_DIR
from predictor.aqua_predictor import AquaSentinelPredictor
from utils.artifacts import ensure_artifacts, load_points

api = Blueprint("api", __name__)

_PREDICTOR = None


def _get_predictor():
    global _PREDICTOR
    ensure_artifacts()
    if _PREDICTOR is None:
        _PREDICTOR = AquaSentinelPredictor(MODEL_PATH)
    return _PREDICTOR


@api.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@api.route("/score", methods=["POST"])
def score():
    payload = request.get_json(silent=True) or {}
    lat = payload.get("lat")
    lon = payload.get("lon")
    date = payload.get("date", datetime.utcnow().strftime("%Y-%m-%d"))
    if lat is None or lon is None:
        return jsonify({"error": "lat and lon are required"}), 400

    predictor = _get_predictor()
    score_val, lower, upper = predictor.predict_with_interval(
        float(lat), float(lon), date
    )
    return jsonify({
        "lat": lat,
        "lon": lon,
        "date": date,
        "score": score_val,
        "interval_lower": lower,
        "interval_upper": upper,
    })


@api.route("/score/batch", methods=["POST"])
def score_batch():
    predictor = _get_predictor()

    if "file" in request.files:
        file = request.files["file"]
        df = pd.read_csv(file)
    else:
        payload = request.get_json(silent=True) or []
        df = pd.DataFrame(payload)

    if df.empty or not {"lat", "lon"}.issubset(df.columns):
        return jsonify({"error": "Provide lat/lon columns or JSON list."}), 400

    if "date" not in df.columns:
        df["date"] = datetime.utcnow().strftime("%Y-%m-%d")

    results = df.apply(
        lambda row: predictor.predict_with_interval(row["lat"], row["lon"], row["date"]),
        axis=1,
    )
    df["score"], df["interval_lower"], df["interval_upper"] = zip(*results)

    output = io.StringIO()
    df.to_csv(output, index=False)
    output.seek(0)

    return send_file(
        io.BytesIO(output.getvalue().encode("utf-8")),
        mimetype="text/csv",
        as_attachment=True,
        download_name="batch_scores.csv",
    )


@api.route("/points", methods=["GET"])
def points():
    limit = request.args.get("limit", type=int)
    start_date = request.args.get("start")
    end_date = request.args.get("end")
    df = load_points(limit=limit, start_date=start_date, end_date=end_date)
    return jsonify(df.to_dict(orient="records"))


@api.route("/export/csv", methods=["GET"])
def export_csv():
    points_path = ensure_artifacts()
    return send_file(points_path, mimetype="text/csv", as_attachment=True)


@api.route("/export/pdf", methods=["GET"])
def export_pdf():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    points_path = ensure_artifacts()
    df = pd.read_csv(points_path)
    report_path = os.path.join(RESULTS_DIR, "model_report.json")
    try:
        with open(report_path, "r", encoding="utf-8") as handle:
            report = json.load(handle)
    except Exception:
        report = {}

    fig, ax = plt.subplots(2, 1, figsize=(8.5, 11))
    ax[0].set_title("Waterborne Disease Risk Score Distribution")
    ax[0].hist(df["risk_score"], bins=20, color="#0f6c79", alpha=0.8)
    ax[0].set_xlabel("Risk Score")
    ax[0].set_ylabel("Count")

    ax[1].axis("off")
    metrics = report.get("metrics", {})
    lines = ["Model Report"]
    for key in ["selected_model", "mae", "rmse", "r2", "interval_coverage", "interval_width"]:
        if key in metrics:
            lines.append(f"{key}: {metrics[key]}")
    ax[1].text(0.05, 0.9, "\n".join(lines), fontsize=12, va="top")

    output = io.BytesIO()
    plt.tight_layout()
    plt.savefig(output, format="pdf")
    plt.close(fig)
    output.seek(0)

    return send_file(output, mimetype="application/pdf", as_attachment=True, download_name="outbreaks_report.pdf")

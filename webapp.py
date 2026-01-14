import os
from datetime import datetime

from flask import Flask, render_template, request

from api.endpoints import api as api_blueprint
from config.settings import MODEL_PATH
from predictor.aqua_predictor import AquaSentinelPredictor
from utils.artifacts import ensure_artifacts


APP_TITLE = "Outbreaks"
APP_TAGLINE = "Preempting Waterborne Disease Outbreaks with AI and Satellite Data"


def create_app():
    app = Flask(
        __name__,
        template_folder=os.path.join(os.path.dirname(__file__), "web", "templates"),
        static_folder=os.path.join(os.path.dirname(__file__), "web", "static"),
    )

    ensure_artifacts()
    predictor = AquaSentinelPredictor(MODEL_PATH)
    app.register_blueprint(api_blueprint, url_prefix="/api")

    @app.route("/", methods=["GET", "POST"])
    def index():
        lat = request.form.get("lat", "0.5")
        lon = request.form.get("lon", "32.5")
        date = request.form.get("date", datetime.utcnow().strftime("%Y-%m-%d"))
        score = None
        interval = None
        threshold = request.form.get("threshold", "70")
        if request.method == "POST":
            try:
                score, lower, upper = predictor.predict_with_interval(
                    float(lat), float(lon), date
                )
                if lower is not None and upper is not None:
                    interval = (lower, upper)
            except ValueError:
                score = None

        return render_template(
            "index.html",
            title=APP_TITLE,
            tagline=APP_TAGLINE,
            lat=lat,
            lon=lon,
            date=date,
            score=score,
            interval=interval,
            threshold=threshold,
        )

    return app


if __name__ == "__main__":
    application = create_app()
    port = int(os.getenv("PORT", "8001"))
    application.run(host="0.0.0.0", port=port, debug=True)

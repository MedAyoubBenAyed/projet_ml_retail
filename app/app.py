"""Flask API exposing churn prediction endpoint."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from flask import Flask, jsonify, request

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from predict import ChurnPredictor


def create_app() -> Flask:
    app = Flask(__name__)

    model_path = os.getenv(
        "CHURN_MODEL_PATH", str(PROJECT_ROOT / "models" / "best_churn_pipeline.joblib")
    )
    predictor = ChurnPredictor(model_path=model_path)

    @app.get("/")
    def index():
        return jsonify({"message": "Retail churn API is running"})

    @app.post("/predict")
    def predict():
        payload = request.get_json(silent=True)
        if payload is None:
            return jsonify({"error": "Request body must be valid JSON"}), 400

        try:
            result = predictor.predict(payload)
            return jsonify(result)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        except Exception as exc:
            return jsonify({"error": f"Prediction failed: {exc}"}), 500

    return app


app = create_app()


if __name__ == "__main__":
    app.run(debug=True)

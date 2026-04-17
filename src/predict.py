"""Model loading and prediction helpers for deployment."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from utils import load_model


DEFAULT_MODEL_PATH = Path("models/best_churn_pipeline.joblib")


class ChurnPredictor:
    """Inference service around the saved churn pipeline."""

    def __init__(self, model_path: str | Path = DEFAULT_MODEL_PATH):
        self.model_path = Path(model_path)
        self.model = load_model(self.model_path)

    def predict(self, customer_payload: dict) -> dict[str, float | int]:
        """Predict churn class and probability for one customer payload."""
        if not isinstance(customer_payload, dict) or not customer_payload:
            raise ValueError("customer_payload must be a non-empty dictionary")

        customer_df = pd.DataFrame([customer_payload])
        expected_features = getattr(self.model, "feature_names_in_", None)
        if expected_features is not None:
            for feature in expected_features:
                if feature not in customer_df.columns:
                    customer_df[feature] = None
            customer_df = customer_df[list(expected_features)]

        prediction = int(self.model.predict(customer_df)[0])

        churn_probability = None
        if hasattr(self.model, "predict_proba"):
            churn_probability = float(self.model.predict_proba(customer_df)[:, 1][0])

        return {
            "churn_prediction": prediction,
            "churn_probability": churn_probability,
        }

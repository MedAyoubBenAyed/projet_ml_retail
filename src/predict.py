"""Prediction helpers for the retail churn project."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import joblib
import pandas as pd

import preprocessing as prep


DEFAULT_MODEL_PATH = Path("models/churn_model_bundle.joblib")
DEFAULT_OUTPUT_PATH = Path("reports/predictions.csv")


def load_bundle(model_path: str | Path) -> Dict[str, Any]:
    """Load the persisted model bundle."""
    bundle = joblib.load(model_path)
    if not isinstance(bundle, dict):
        raise TypeError("Unexpected model artifact format.")
    return bundle


def prepare_features(df: pd.DataFrame, bundle: Dict[str, Any]) -> pd.DataFrame:
    """Apply the same preprocessing chain used at training time."""
    df = prep.prepare_dataframe(df, require_target=False)

    input_columns = bundle["input_columns"]
    df = df.reindex(columns=input_columns)

    preprocessor = bundle["preprocessor"]
    transformed = preprocessor.transform(df)

    feature_names = bundle.get("feature_names_out") or []
    if feature_names and len(feature_names) == transformed.shape[1]:
        features = pd.DataFrame(transformed, columns=feature_names, index=df.index)
    else:
        features = pd.DataFrame(transformed, index=df.index)

    final_columns = bundle["final_columns"]
    features = features.reindex(columns=final_columns, fill_value=0.0)
    return features


def predict_dataframe(df: pd.DataFrame, bundle: Dict[str, Any]) -> pd.DataFrame:
    """Return a DataFrame containing predictions and optional probabilities."""
    features = prepare_features(df, bundle)
    model = bundle["model"]

    output = pd.DataFrame(index=df.index)
    output["prediction"] = model.predict(features)

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features)[:, 1]
        output["probability"] = probabilities

    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict churn on a CSV file")
    parser.add_argument("--input-path", type=str, required=True)
    parser.add_argument("--model-path", type=str, default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--output-path", type=str, default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--no-save", action="store_true", help="Do not save predictions to disk")
    args = parser.parse_args()

    input_path = Path(args.input_path)
    model_path = Path(args.model_path)
    output_path = Path(args.output_path)

    bundle = load_bundle(model_path)
    df_input = pd.read_csv(input_path)
    predictions = predict_dataframe(df_input, bundle)

    result = pd.concat([df_input.reset_index(drop=True), predictions.reset_index(drop=True)], axis=1)

    if not args.no_save:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(output_path, index=False)
        print(f"Predictions saved to: {output_path}")

    print(result.head().to_string())


if __name__ == "__main__":
    main()

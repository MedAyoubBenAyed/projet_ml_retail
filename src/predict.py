"""Prediction helpers for the retail churn project."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import joblib
import pandas as pd

import preprocessing as prep


DEFAULT_MODEL_PATH = Path(__file__).parent.parent / "models" / "churn_model_bundle.joblib"
DEFAULT_OUTPUT_PATH = Path(__file__).parent.parent / "reports" / "predictions.csv"


def load_bundle(model_path: str | Path) -> Dict[str, Any]:
    """Load the persisted model bundle."""
    bundle = joblib.load(model_path)
    if not isinstance(bundle, dict):
        raise TypeError("Unexpected model artifact format.")
    return bundle


def prepare_features(df: pd.DataFrame, bundle: Dict[str, Any]) -> pd.DataFrame:
    """Apply the same preprocessing chain used at training time."""
    df = prep.prepare_dataframe(df, require_target=False)

    preprocessor = bundle["preprocessor"]
    if preprocessor is None:
        raise ValueError("Preprocessor not found in model bundle. Re-train the model.")
    
    transformed = preprocessor.transform(df)

    # Recréer un DataFrame avec les noms des features du préprocesseur, si possible.
    try:
        feature_names = preprocessor.get_feature_names_out()
        features = pd.DataFrame(transformed, index=df.index, columns=feature_names)
    except Exception:
        # Fallback: pas de noms disponibles (ancien sklearn / pipeline)
        features = pd.DataFrame(transformed, index=df.index)

    # Appliquer exactement le même sous-ensemble de features conservées à l'entraînement.
    # On réaligne strictement les colonnes pour éviter les écarts train/inférence.
    kept = bundle.get("kept_feature_names") or bundle.get("final_columns") or []
    if kept:
        # Format attendu : noms de colonnes conservées après sélection (corr/VIF/leakage).
        if isinstance(kept[0], str):
            expected_cols = [str(col) for col in kept]
            # Reindex garantit:
            # - même ordre de colonnes qu'au fit
            # - suppression des colonnes en trop
            # - ajout des colonnes manquantes avec 0.0
            features = features.reindex(columns=expected_cols, fill_value=0.0)
        else:
            # Ancien format: indices numériques (0..k-1).
            try:
                kept_int = [int(c) if isinstance(c, str) and str(c).isdigit() else int(c) for c in kept]
                features = features.iloc[:, kept_int]
            except Exception:
                # Si on ne peut pas aligner proprement, on laisse toutes les colonnes
                pass

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


def print_prediction_summary(result: pd.DataFrame) -> None:
    """Print a compact, readable prediction summary for the terminal."""
    preview_columns = [
        col for col in ["CustomerID", "Churn", "prediction", "probability"]
        if col in result.columns
    ]

    print("\n=== Prediction summary ===")
    print(f"Rows: {len(result)}")

    if "prediction" in result.columns:
        counts = result["prediction"].value_counts().sort_index()
        print("Prediction counts:")
        for label, count in counts.items():
            print(f"  {label}: {count}")

    if "probability" in result.columns:
        print(f"Average probability: {result['probability'].mean():.4f}")

    if preview_columns:
        print("\nPreview:")
        print(result.loc[:, preview_columns].head().to_string(index=False))
    else:
        print("\nPreview:")
        print(result.head().to_string(index=False))


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

    print_prediction_summary(result)


if __name__ == "__main__":
    main()
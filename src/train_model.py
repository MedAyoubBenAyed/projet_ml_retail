"""Training script for the retail churn classification project."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

import preprocessing as prep


DEFAULT_MODEL_PATH = Path("models/churn_model_bundle.joblib")
DEFAULT_METRICS_PATH = Path("models/training_metrics.json")
DEFAULT_OUTPUT_DIR = Path("data/train_test")
DEFAULT_RAW_PATH = prep.DEFAULT_RAW_PATH
TARGET_COL = prep.TARGET_COL


def build_model(model_type: str, random_state: int) -> Any:
    """Create the classifier used for churn prediction."""
    if model_type == "random_forest":
        return RandomForestClassifier(
            n_estimators=300,
            random_state=random_state,
            class_weight="balanced_subsample",
            n_jobs=-1,
        )

    return LogisticRegression(
        max_iter=3000,
        class_weight="balanced",
        solver="lbfgs",
    )


def evaluate_model(model: Any, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
    """Compute a compact set of evaluation metrics."""
    y_pred = model.predict(X_test)
    metrics: Dict[str, Any] = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(y_test, y_pred, zero_division=0),
    }

    if hasattr(model, "predict_proba"):
        try:
            y_prob = model.predict_proba(X_test)[:, 1]
            metrics["roc_auc"] = float(roc_auc_score(y_test, y_prob))
        except Exception:
            metrics["roc_auc"] = None
    else:
        metrics["roc_auc"] = None

    return metrics


def fit_training_artifacts(
    df: pd.DataFrame,
    model_type: str,
    test_size: float,
    random_state: int,
    corr_threshold: float,
    vif_threshold: float,
) -> Tuple[Dict[str, Any], pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Fit preprocessing and the classifier, then return the artifacts."""
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    numeric_cols, ordinal_cols, nominal_cols = prep.infer_feature_groups(X_train)
    preprocessor = prep.build_preprocessor(numeric_cols, ordinal_cols, nominal_cols)

    X_train_arr = preprocessor.fit_transform(X_train)
    X_test_arr = preprocessor.transform(X_test)

    feature_names = prep.get_feature_names(preprocessor)
    if feature_names and len(feature_names) == X_train_arr.shape[1]:
        X_train_df = pd.DataFrame(X_train_arr, columns=feature_names, index=X_train.index)
        X_test_df = pd.DataFrame(X_test_arr, columns=feature_names, index=X_test.index)
    else:
        X_train_df = pd.DataFrame(X_train_arr, index=X_train.index)
        X_test_df = pd.DataFrame(X_test_arr, index=X_test.index)

    X_train_df, X_test_df, dropped_corr = prep.remove_correlated_features(
        X_train_df,
        X_test_df,
        threshold=corr_threshold,
    )
    X_train_df, X_test_df, dropped_vif = prep.remove_high_vif_features(
        X_train_df,
        X_test_df,
        vif_threshold=vif_threshold,
    )

    model = build_model(model_type=model_type, random_state=random_state)
    model.fit(X_train_df, y_train)

    metrics = evaluate_model(model, X_test_df, y_test)

    artifact: Dict[str, Any] = {
        "model": model,
        "preprocessor": preprocessor,
        "feature_names_out": feature_names,
        "input_columns": X_train.columns.tolist(),
        "final_columns": X_train_df.columns.tolist(),
        "numeric_cols": numeric_cols,
        "ordinal_cols": ordinal_cols,
        "nominal_cols": nominal_cols,
        "dropped_corr": dropped_corr,
        "dropped_vif": dropped_vif,
        "target_col": TARGET_COL,
        "model_type": model_type,
        "random_state": random_state,
        "metrics": metrics,
    }

    return artifact, X_train_df, X_test_df, y_train, y_test


def save_outputs(
    artifact: Dict[str, Any],
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    output_dir: Path,
    model_path: Path,
    metrics_path: Path,
) -> None:
    """Persist the train/test splits, the bundle, and the metrics."""
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    X_train.to_csv(output_dir / "X_train.csv", index=False)
    X_test.to_csv(output_dir / "X_test.csv", index=False)
    y_train.to_frame(name=TARGET_COL).to_csv(output_dir / "y_train.csv", index=False)
    y_test.to_frame(name=TARGET_COL).to_csv(output_dir / "y_test.csv", index=False)

    joblib.dump(artifact, model_path)
    metrics_path.write_text(
        json.dumps(artifact["metrics"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the retail churn model")
    parser.add_argument("--raw-path", type=str, default=str(DEFAULT_RAW_PATH))
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--model-path", type=str, default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--metrics-path", type=str, default=str(DEFAULT_METRICS_PATH))
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["logistic", "random_forest"],
        default="logistic",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--corr-threshold", type=float, default=0.90)
    parser.add_argument("--vif-threshold", type=float, default=10.0)
    parser.add_argument("--random-state", type=int, default=prep.RANDOM_STATE)
    args = parser.parse_args()

    raw_path = Path(args.raw_path)
    output_dir = Path(args.output_dir)
    model_path = Path(args.model_path)
    metrics_path = Path(args.metrics_path)

    df_raw = prep.load_raw_dataset(raw_path)
    df_prepared = prep.prepare_dataframe(df_raw)

    artifact, X_train, X_test, y_train, y_test = fit_training_artifacts(
        df=df_prepared,
        model_type=args.model_type,
        test_size=args.test_size,
        random_state=args.random_state,
        corr_threshold=args.corr_threshold,
        vif_threshold=args.vif_threshold,
    )

    save_outputs(
        artifact=artifact,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        output_dir=output_dir,
        model_path=model_path,
        metrics_path=metrics_path,
    )

    print("Training finished.")
    print(f"Model bundle saved to: {model_path}")
    print(f"Train/test splits saved to: {output_dir}")
    print(f"Metrics saved to: {metrics_path}")
    print(json.dumps(artifact["metrics"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

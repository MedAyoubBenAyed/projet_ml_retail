"""Training script for the retail churn classification project."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


import preprocessing as prep
try:
    from imblearn.over_sampling import SMOTE
except Exception:
    SMOTE = None


DEFAULT_MODEL_PATH = Path("models/churn_model_bundle.joblib")
DEFAULT_METRICS_PATH = Path("models/training_metrics.json")
DEFAULT_OUTPUT_DIR = Path("data/train_test")
# preprocessing.py provides `TARGET` and `RANDOM_STATE` constants and a `run()` entrypoint.
# Use the raw CSV present in the repo as the default raw path.
DEFAULT_RAW_PATH = Path("data/raw/retail_customers_COMPLETE_CATEGORICAL.csv")
TARGET_COL = prep.TARGET


def build_reference_row(df_prepared: pd.DataFrame) -> Dict[str, Any]:
    """
    Build a deterministic "typical customer" row used for single-row inference.
    Numeric columns -> median, categorical -> mode, others -> first non-null.
    """
    X = df_prepared.drop(columns=[TARGET_COL], errors="ignore")
    ref: Dict[str, Any] = {}

    for col in X.columns:
        s = X[col]
        if pd.api.types.is_numeric_dtype(s):
            val = pd.to_numeric(s, errors="coerce").median(skipna=True)
            ref[col] = None if pd.isna(val) else float(val)
        else:
            mode = s.dropna().mode()
            if len(mode) > 0:
                ref[col] = mode.iloc[0]
            else:
                # last resort: keep None (will be imputed by preprocessor)
                ref[col] = None

    return ref


def build_model(model_type: str, random_state: int) -> Any:
    """Create the classifier used for churn prediction."""
    if model_type == "random_forest":
        return RandomForestClassifier(
            n_estimators=300,
            random_state=random_state,
            class_weight="balanced_subsample",
            n_jobs=-1,
        )

    elif model_type == "svc":
        return SVC(
            probability=True,
            class_weight="balanced"
        )

    else:  # logistic
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


def print_metrics(metrics: Dict[str, Any]) -> None:
    """Print training metrics in a readable format."""
    print("\n" + "=" * 60)
    print("TRAINING METRICS")
    print("=" * 60)
    print(f"Accuracy:   {metrics['accuracy']:.4f}")
    print(f"Precision:  {metrics['precision']:.4f}")
    print(f"Recall:     {metrics['recall']:.4f}")
    print(f"F1-Score:   {metrics['f1']:.4f}")
    if metrics.get("roc_auc") is not None:
        print(f"ROC AUC:    {metrics['roc_auc']:.4f}")
    
    print("\nConfusion Matrix:")
    cm = metrics["confusion_matrix"]
    print(f"  [[{cm[0][0]:5d}, {cm[0][1]:5d}],")
    print(f"   [{cm[1][0]:5d}, {cm[1][1]:5d}]]")
    
    print("\nClassification Report:")
    print(metrics["classification_report"])
    print("=" * 60 + "\n")





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
        choices=["logistic", "random_forest", "svc"],
        default="logistic",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--corr-threshold", type=float, default=0.90)
    parser.add_argument("--vif-threshold", type=float, default=10.0)
    parser.add_argument("--enable-vif", action="store_true")
    parser.add_argument("--leakage-threshold", type=float, default=0.3)
    parser.add_argument("--random-state", type=int, default=prep.RANDOM_STATE)
    args = parser.parse_args()

    raw_path = Path(args.raw_path)
    output_dir = Path(args.output_dir)
    model_path = Path(args.model_path)
    metrics_path = Path(args.metrics_path)

    # Load raw dataset and run the preprocessing pipeline from `preprocessing.py`.
    # The preprocessing pipeline saves the train/test splits to `output_dir`.
    df_raw = prep.load_raw_dataset(raw_path)
    df_prepared = prep.prepare_dataframe(df_raw)
    reference_row = build_reference_row(df_prepared)
    raw_feature_columns = df_prepared.drop(columns=[TARGET_COL], errors="ignore").columns.tolist()
    preprocessor, final_columns = prep.split_and_transform(
        df=df_prepared,
        output_dir=output_dir,
        processed_path=Path("data/processed/retail_customers_processed.csv"),
        test_size=args.test_size,
        random_state=args.random_state,
        corr_threshold=args.corr_threshold,
        vif_threshold=args.vif_threshold,
        enable_vif=args.enable_vif,
        leakage_threshold=args.leakage_threshold,
    )

    # Load the preprocessed train/test splits produced by `prep.split_and_transform()`.
    X_train = pd.read_csv(output_dir / "X_train.csv")
    X_test = pd.read_csv(output_dir / "X_test.csv")
    y_train = pd.read_csv(output_dir / "y_train.csv")[TARGET_COL]
    y_test = pd.read_csv(output_dir / "y_test.csv")[TARGET_COL]

    # Build, fit and evaluate the model on the preprocessed splits.
    model = build_model(model_type=args.model_type, random_state=args.random_state)

    # Try SMOTE; if unavailable, fall back to simple random oversampling
    sampling_applied = None
    if SMOTE is not None:
        try:
            print("[SMOTE] Applying SMOTE to training set...")
            print("[SMOTE] Original class distribution:\n", y_train.value_counts().to_dict())
            sm = SMOTE(random_state=args.random_state)
            X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
            print("[SMOTE] Resampled class distribution:\n", y_train_res.value_counts().to_dict())
            X_train = X_train_res
            y_train = y_train_res
            sampling_applied = "smote"
        except Exception as e:
            print(f"[SMOTE] Failed to apply SMOTE: {e}")

    if sampling_applied is None:
        # Simple random oversampling of minority class to match majority
        try:
            vc = y_train.value_counts()
            if len(vc) > 1 and vc.max() != vc.min():
                majority_class = vc.idxmax()
                majority_count = int(vc.max())
                parts = []
                for cls, count in vc.items():
                    cls_idx = (y_train == cls)
                    X_cls = X_train[cls_idx]
                    y_cls = y_train[cls_idx]
                    if count < majority_count:
                        # resample with replacement
                        X_resampled = X_cls.sample(majority_count, replace=True, random_state=args.random_state)
                        y_resampled = pd.Series([cls] * majority_count, index=X_resampled.index)
                    else:
                        X_resampled = X_cls.sample(majority_count, replace=False, random_state=args.random_state)
                        y_resampled = pd.Series([cls] * majority_count, index=X_resampled.index)
                    parts.append((X_resampled, y_resampled))

                X_new = pd.concat([p[0] for p in parts], axis=0)
                y_new = pd.concat([p[1] for p in parts], axis=0)
                # shuffle
                shuffled_idx = X_new.sample(frac=1, random_state=args.random_state).index
                X_train = X_new.loc[shuffled_idx].reset_index(drop=True)
                y_train = y_new.loc[shuffled_idx].reset_index(drop=True)
                sampling_applied = "random_oversample"
                print("[OVERSAMPLE] Applied simple random oversampling. New class distribution:\n", y_train.value_counts().to_dict())
            else:
                print("[OVERSAMPLE] Training set already balanced; no sampling applied.")
        except Exception as e:
            print(f"[OVERSAMPLE] Failed to apply random oversampling: {e}")

    model.fit(X_train, y_train)
    metrics = evaluate_model(model, X_test, y_test)

    artifact = {
        "model": model,
        "preprocessor": preprocessor,
        # For deterministic single-row inference (UI): build a full row then override fields.
        "raw_feature_columns": raw_feature_columns,
        "reference_row": reference_row,
        # X_train columns correspond to the final, kept feature set after all dropping steps.
        "feature_names_out": X_train.columns.tolist() if hasattr(X_train, "columns") else None,
        "input_columns": X_train.columns.tolist() if hasattr(X_train, "columns") else [],
        # Persist explicit kept feature names for inference-time alignment.
        "final_columns": X_train.columns.tolist() if hasattr(X_train, "columns") else final_columns,
        "kept_feature_names": X_train.columns.tolist() if hasattr(X_train, "columns") else final_columns,
        "numeric_cols": None,
        "ordinal_cols": None,
        "nominal_cols": None,
        "dropped_corr": None,
        "dropped_vif": None,
        "dropped_leakage": None,
        "target_col": TARGET_COL,
        "model_type": args.model_type,
        "random_state": args.random_state,
        "sampling_applied": sampling_applied,
        "metrics": metrics,
    }

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

    print("\nTraining finished successfully.")
    print(f"  Model bundle: {model_path}")
    print(f"  Train/test splits: {output_dir}")
    print(f"  Metrics file: {metrics_path}")
    
    print_metrics(artifact["metrics"])


if __name__ == "__main__":
    main()

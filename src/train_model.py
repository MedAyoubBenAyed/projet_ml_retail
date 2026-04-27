"""Training entrypoint for churn model comparison and model persistence."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from preprocessing import create_training_pipeline, get_fitted_pca_component_count
from utils import evaluate_model, save_json, save_model


DEFAULT_DATA_PATH = Path("data/raw/retail_customers_COMPLETE_CATEGORICAL.csv")
DEFAULT_MODEL_PATH = Path("models/best_churn_pipeline.joblib")
DEFAULT_REPORT_PATH = Path("reports/model_metrics.json")
DEFAULT_SPLIT_DIR = Path("data/train_test")


def _build_model_candidates() -> dict[str, object]:
    models = {
        "logistic_regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        "knn": KNeighborsClassifier(n_neighbors=9),
    }

    try:
        from xgboost import XGBClassifier

        models["xgboost"] = XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="binary:logistic",
            eval_metric="auc",
            random_state=42,
        )
    except ImportError:
        print("XGBoost not installed: training will continue without xgboost.")

    return models


def train_and_select_best_model(
    dataset_path: str | Path = DEFAULT_DATA_PATH,
    model_output_path: str | Path = DEFAULT_MODEL_PATH,
    report_output_path: str | Path = DEFAULT_REPORT_PATH,
    split_output_dir: str | Path = DEFAULT_SPLIT_DIR,
    target_column: str = "Churn",
    pca_variance_threshold: float = 0.95,
    test_size: float = 0.2,
):
    """Train models, compare metrics, and save best fitted pipeline."""
    df = pd.read_csv(dataset_path)
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataset.")

    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Use dynamic routing in preprocessing so engineered columns are retained.
    numerical_features = None
    categorical_features = None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=42,
        stratify=y,
    )

    split_dir = Path(split_output_dir)
    split_dir.mkdir(parents=True, exist_ok=True)
    X_train.to_csv(split_dir / "X_train.csv", index=False)
    X_test.to_csv(split_dir / "X_test.csv", index=False)
    y_train.to_frame(name=target_column).to_csv(split_dir / "y_train.csv", index=False)
    y_test.to_frame(name=target_column).to_csv(split_dir / "y_test.csv", index=False)

    model_candidates = _build_model_candidates()

    metrics_by_model = {}
    best_model_name = None
    best_model = None
    best_f1 = -1.0

    for model_name, model in model_candidates.items():
        pipeline = create_training_pipeline(
            numerical_features=numerical_features,
            categorical_features=categorical_features,
            classifier=model,
            pca_variance_threshold=pca_variance_threshold,
            use_smote=True,
        )
        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        y_proba = None
        if hasattr(pipeline, "predict_proba"):
            y_proba = pipeline.predict_proba(X_test)[:, 1]

        metrics = evaluate_model(y_true=y_test, y_pred=y_pred, y_proba=y_proba)
        metrics["pca_components"] = get_fitted_pca_component_count(pipeline)
        metrics_by_model[model_name] = metrics

        if metrics["f1_score"] > best_f1:
            best_f1 = metrics["f1_score"]
            best_model_name = model_name
            best_model = pipeline

    if best_model is None:
        raise RuntimeError("No model could be trained.")

    Path(model_output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(report_output_path).parent.mkdir(parents=True, exist_ok=True)
    save_model(best_model, model_output_path)
    save_json(
        {
            "best_model": best_model_name,
            "selection_metric": "f1_score",
            "metrics": metrics_by_model,
        },
        report_output_path,
    )

    return best_model_name, metrics_by_model


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train churn models with PCA pipeline")
    parser.add_argument("--data", default=str(DEFAULT_DATA_PATH), help="Input CSV path")
    parser.add_argument(
        "--split-output-dir",
        default=str(DEFAULT_SPLIT_DIR),
        help="Directory where train/test CSV split files are exported",
    )
    parser.add_argument(
        "--model-output",
        default=str(DEFAULT_MODEL_PATH),
        help="Output path for best model",
    )
    parser.add_argument(
        "--report-output",
        default=str(DEFAULT_REPORT_PATH),
        help="Output path for model comparison report",
    )
    parser.add_argument("--target", default="Churn", help="Target column name")
    parser.add_argument(
        "--pca-variance",
        type=float,
        default=0.95,
        help="Explained variance threshold for PCA (0,1]",
    )
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    best_model_name, metrics = train_and_select_best_model(
        dataset_path=args.data,
        model_output_path=args.model_output,
        report_output_path=args.report_output,
        split_output_dir=args.split_output_dir,
        target_column=args.target,
        pca_variance_threshold=args.pca_variance,
    )

    print(f"Best model: {best_model_name}")
    for name, values in metrics.items():
        print(
            f"- {name}: f1={values['f1_score']:.4f}, "
            f"roc_auc={values.get('roc_auc', float('nan')):.4f}, "
            f"pca_components={values.get('pca_components')}"
        )


if __name__ == "__main__":
    main()

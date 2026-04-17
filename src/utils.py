"""Shared utility helpers for training, evaluation and persistence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_model(y_true, y_pred, y_proba=None) -> dict[str, Any]:
    """Compute standard binary-classification metrics."""
    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(
            y_true, y_pred, zero_division=0, output_dict=True
        ),
    }

    if y_proba is not None:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))

    return metrics


def save_model(model, model_path: str | Path) -> None:
    """Persist model on disk with joblib."""
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)


def load_model(model_path: str | Path):
    """Load a joblib serialized model from disk."""
    return joblib.load(model_path)


def save_json(payload: dict[str, Any], output_path: str | Path) -> None:
    """Save JSON report to disk."""
    def _json_default(value):
        if hasattr(value, "item"):
            return value.item()
        raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, default=_json_default), encoding="utf-8"
    )

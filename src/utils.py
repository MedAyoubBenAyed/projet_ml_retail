"""Utility helpers for the project."""
from __future__ import annotations

from pathlib import Path
import pandas as pd
import joblib


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)


def save_csv(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def save_model(obj, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)


def load_model(path: str | Path):
    return joblib.load(path)
"""Preprocessing and leakage-safe training pipeline utilities.

This module centralizes feature cleaning + transformation using rules
validated in the exploration notebook, while keeping a production-safe
pipeline for train/test workflows.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler


COLS_HIGH_CORR = [
    "MonetaryMin",
    "MonetaryMax",
    "TotalQuantity",
    "MinQuantity",
    "MaxQuantity",
    "UniqueDescriptions",
    "UniqueDesc",
    "CancelledTransactions",
    "CancelledTrans",
    "TotalTransactions",
    "TotalTrans",
    "UniqueInvoices",
    "AvgLinesPerInvoice",
    "AvgLinesPerInv",
]
COLS_HIGH_VIF = ["CustomerID", "CustomerTenureDays", "FirstPurchaseDaysAgo"]
COLS_USELESS = [
    "NewsletterSubscribed",
    "Newsletter",
    "LastLoginIP",
    # Direct/near-direct target proxies that can leak churn label.
    "ChurnRiskCategory",
    "ChurnRisk",
]
COLS_TO_DROP = list(dict.fromkeys(COLS_HIGH_CORR + COLS_HIGH_VIF + COLS_USELESS))

ORDINAL_MAPPINGS = {
    "SpendingCategory": ["Low", "Medium", "High", "VIP"],
    "SpendingCat": ["Low", "Medium", "High", "VIP"],
    "AgeCategory": ["18-24", "25-34", "35-44", "45-54", "55-64", "65+", "Inconnu"],
    "LoyaltyLevel": ["Nouveau", "Jeune", "Etabli", "Ancien", "Inconnu"],
    "ChurnRiskCategory": ["Faible", "Moyen", "Eleve", "Critique"],
    "ChurnRisk": ["Faible", "Moyen", "Eleve", "Critique"],
    "BasketSizeCategory": ["Petit", "Moyen", "Grand", "Inconnu"],
    "BasketSize": ["Petit", "Moyen", "Grand", "Inconnu"],
    "PreferredTimeOfDay": ["Matin", "Midi", "Apres-midi", "Soir", "Nuit"],
    "PreferredTime": ["Matin", "Midi", "Apres-midi", "Soir", "Nuit"],
}

NOMINAL_COLS = [
    "RFMSegment",
    "CustomerType",
    "FavoriteSeason",
    "Region",
    "WeekendPreference",
    "WeekendPref",
    "ProductDiversity",
    "ProdDiversity",
    "Gender",
    "AccountStatus",
    "Country",
]


def _make_one_hot_encoder() -> OneHotEncoder:
    """Build OneHotEncoder compatible with multiple scikit-learn versions."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


class DataCleaner(BaseEstimator, TransformerMixin):
    """Apply deterministic cleaning/feature-engineering rules before encoding."""

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        df = pd.DataFrame(X).copy() if not isinstance(X, pd.DataFrame) else X.copy()

        registration_series = None
        if "RegistrationDate" in df.columns:
            registration_series = df["RegistrationDate"].copy()

        df = df.drop(columns=[c for c in COLS_TO_DROP if c in df.columns], errors="ignore")

        if "SupportTicketsCount" in df.columns:
            df["SupportTicketsCount"] = df["SupportTicketsCount"].replace({-1: np.nan, 999: np.nan})
        if "SupportTickets" in df.columns:
            df["SupportTickets"] = df["SupportTickets"].replace({-1: np.nan, 999: np.nan})

        if "SatisfactionScore" in df.columns:
            df["SatisfactionScore"] = df["SatisfactionScore"].replace({-1: np.nan, 99: np.nan, 0: np.nan})
        if "Satisfaction" in df.columns:
            df["Satisfaction"] = df["Satisfaction"].replace({-1: np.nan, 99: np.nan, 0: np.nan})

        if registration_series is not None:
            reg = pd.to_datetime(registration_series, dayfirst=True, errors="coerce")
            df["RegYear"] = reg.dt.year.astype("float64")
            df["RegMonth"] = reg.dt.month.astype("float64")
            df["RegDay"] = reg.dt.day.astype("float64")
            df["RegWeekday"] = reg.dt.weekday.astype("float64")

        if "MonetaryTotal" in df.columns and "Recency" in df.columns:
            df["MonetaryPerDay"] = df["MonetaryTotal"] / (df["Recency"] + 1)

        if "MonetaryTotal" in df.columns and "Frequency" in df.columns:
            freq = df["Frequency"].replace(0, np.nan)
            df["AvgBasketValue"] = df["MonetaryTotal"] / freq

        return df


class _DynamicColumnTransformer(BaseEstimator, TransformerMixin):
    """Infer available columns after cleaning and route them to sub-pipelines."""

    def __init__(
        self,
        numerical_pipeline,
        ordinal_pipeline,
        nominal_pipeline,
        allowed_numerical: Sequence[str] | None = None,
        allowed_categorical: Sequence[str] | None = None,
    ):
        self.numerical_pipeline = numerical_pipeline
        self.ordinal_pipeline = ordinal_pipeline
        self.nominal_pipeline = nominal_pipeline
        self.allowed_numerical = allowed_numerical
        self.allowed_categorical = allowed_categorical

    def _split_columns(self, X):
        ordinal = [c for c in ORDINAL_MAPPINGS if c in X.columns]
        nominal = [c for c in NOMINAL_COLS if c in X.columns and c not in ordinal]

        extra_cat = [
            c
            for c in X.select_dtypes(exclude=[np.number]).columns
            if c not in ordinal and c not in nominal
        ]
        nominal = nominal + extra_cat

        numeric = [c for c in X.select_dtypes(include=[np.number]).columns if c != "Churn"]

        if self.allowed_numerical is not None:
            allowed_num = set(self.allowed_numerical)
            numeric = [c for c in numeric if c in allowed_num]

        if self.allowed_categorical is not None:
            allowed_cat = set(self.allowed_categorical)
            ordinal = [c for c in ordinal if c in allowed_cat]
            nominal = [c for c in nominal if c in allowed_cat]

        return numeric, ordinal, nominal

    def fit(self, X, y=None):
        numeric, ordinal, nominal = self._split_columns(X)

        transformers = []
        if numeric:
            transformers.append(("num", self.numerical_pipeline, numeric))
        if ordinal:
            categories = [ORDINAL_MAPPINGS[c] for c in ordinal]
            self.ordinal_pipeline.named_steps["encoder"].categories = categories
            transformers.append(("ord", self.ordinal_pipeline, ordinal))
        if nominal:
            transformers.append(("nom", self.nominal_pipeline, nominal))

        if not transformers:
            raise ValueError("No columns available for preprocessing after cleaning.")

        self.ct_ = ColumnTransformer(transformers=transformers, remainder="drop")
        self.ct_.fit(X, y)
        return self

    def transform(self, X, y=None):
        return self.ct_.transform(X)


def create_preprocessor(
    numerical_features: Sequence[str] | None = None,
    categorical_features: Sequence[str] | None = None,
    pca_variance_threshold: float = 0.95,
) -> Pipeline:
    """Create notebook-aligned preprocessing pipeline.

    Args kept for backward compatibility with callers that still pass explicit
    feature lists. The cleaner + dynamic transformer infer effective columns
    from input data at fit time.
    """
    if not 0 < pca_variance_threshold <= 1:
        raise ValueError(
            "pca_variance_threshold must be greater than 0 and less than or equal to 1."
        )

    numerical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "pca",
                PCA(
                    n_components=pca_variance_threshold,
                    svd_solver="full",
                    random_state=42,
                ),
            ),
        ]
    )

    ordinal_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OrdinalEncoder(
                    categories=[v for v in ORDINAL_MAPPINGS.values()],
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
            ),
        ]
    )

    nominal_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", _make_one_hot_encoder()),
        ]
    )

    return Pipeline(
        steps=[
            ("cleaner", DataCleaner()),
            (
                "preprocessor",
                _DynamicColumnTransformer(
                    numerical_pipeline=numerical_pipeline,
                    ordinal_pipeline=ordinal_pipeline,
                    nominal_pipeline=nominal_pipeline,
                    allowed_numerical=numerical_features,
                    allowed_categorical=categorical_features,
                ),
            ),
        ]
    )


def create_training_pipeline(
    numerical_features: Sequence[str] | None,
    categorical_features: Sequence[str] | None,
    classifier,
    pca_variance_threshold: float = 0.95,
    use_smote: bool = True,
) -> ImbPipeline:
    """Create complete training pipeline with optional SMOTE and classifier."""
    preprocessor = create_preprocessor(
        numerical_features=numerical_features,
        categorical_features=categorical_features,
        pca_variance_threshold=pca_variance_threshold,
    )

    # imbalanced-learn >=0.14 rejects an intermediate nested sklearn Pipeline.
    # Flatten cleaner + dynamic preprocessor into the top-level imblearn pipeline.
    steps = list(preprocessor.steps)
    if use_smote:
        steps.append(("smote", SMOTE(random_state=42)))
    steps.append(("classifier", classifier))

    return ImbPipeline(steps=steps)


def get_fitted_pca_component_count(fitted_pipeline: ImbPipeline) -> int | None:
    """Return fitted PCA component count from a fitted training pipeline."""
    dynamic_ct = fitted_pipeline.named_steps.get("preprocessor")
    if dynamic_ct is None:
        return None

    try:
        if hasattr(dynamic_ct, "ct_"):
            num_pipe = None
            for name, transformer, _cols in dynamic_ct.ct_.transformers_:
                if name == "num":
                    num_pipe = transformer
                    break
            if num_pipe is None:
                return None
            pca = num_pipe.named_steps.get("pca")
            return getattr(pca, "n_components_", None)
    except Exception:
        return None

    return None

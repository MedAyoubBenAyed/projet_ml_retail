"""Preprocessing and leakage-safe training pipeline utilities."""

from __future__ import annotations

from typing import Sequence

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def create_preprocessor(
    numerical_features: Sequence[str],
    categorical_features: Sequence[str],
    pca_variance_threshold: float = 0.95,
) -> ColumnTransformer:
    """Create a preprocessing transformer.

    Numerical columns: median imputation -> scaling -> PCA.
    Categorical columns: most-frequent imputation -> one-hot encoding.
    """
    if not 0 < pca_variance_threshold <= 1:
        raise ValueError(
            "pca_variance_threshold must be greater than 0 and less than or equal to 1."
        )

    transformers = []

    if numerical_features:
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
        transformers.append(("num", numerical_pipeline, list(numerical_features)))

    if categorical_features:
        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore")),
            ]
        )
        transformers.append(("cat", categorical_pipeline, list(categorical_features)))

    if not transformers:
        raise ValueError("At least one feature must be provided.")

    return ColumnTransformer(transformers=transformers, remainder="drop")


def create_training_pipeline(
    numerical_features: Sequence[str],
    categorical_features: Sequence[str],
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

    steps = [("preprocessor", preprocessor)]
    if use_smote:
        steps.append(("smote", SMOTE(random_state=42)))
    steps.append(("classifier", classifier))

    return ImbPipeline(steps=steps)


def get_fitted_pca_component_count(fitted_pipeline: ImbPipeline) -> int | None:
    """Return fitted PCA component count from a fitted training pipeline."""
    preprocessor = fitted_pipeline.named_steps.get("preprocessor")
    if preprocessor is None or "num" not in preprocessor.named_transformers_:
        return None

    pca = preprocessor.named_transformers_["num"].named_steps.get("pca")
    return getattr(pca, "n_components_", None)

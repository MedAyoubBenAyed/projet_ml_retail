"""Preprocessing helpers for the project."""
from __future__ import annotations

import argparse
import ipaddress
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler


# -----------------------------
# Config
# -----------------------------

DEFAULT_RAW_PATH = Path("data/raw/retail_customers_COMPLETE_CATEGORICAL.csv")
DEFAULT_OUTPUT_DIR = Path("data/train_test")
DEFAULT_PROCESSED_PATH = Path("data/processed/retail_customers_processed.csv")
TARGET_COL = "Churn"
RANDOM_STATE = 42



# -----------------------------
# Feature engineering helpers
# -----------------------------

def parse_registration_date(df: pd.DataFrame, col: str = "RegistrationDate") -> pd.DataFrame:
    """
    Parse RegistrationDate with mixed formats (UK/ISO/US) and extract date parts.
    Drops original column after extraction.
    """
    if col not in df.columns:
        return df

    dt = pd.to_datetime(df[col], dayfirst=True, format="mixed", errors="coerce")

    df = df.copy()
    df["RegYear"]    = dt.dt.year
    df["RegMonth"]   = dt.dt.month
    df["RegDay"]     = dt.dt.day
    df["RegWeekday"] = dt.dt.weekday

    df.drop(columns=[col], inplace=True)
    return df


def featurize_ip(df: pd.DataFrame, col: str = "LastLoginIP") -> pd.DataFrame:
    """
    Minimal IP feature engineering:
    - is_private_ip (0/1)
    - ip_first_octet (0-255) numeric
    Drops original IP column after extraction.
    """
    if col not in df.columns:
        return df

    def _is_private(ip_str: str) -> float:
        try:
            ip = ipaddress.ip_address(str(ip_str))
            return float(ip.is_private)
        except Exception:
            return np.nan

    def _first_octet(ip_str: str) -> float:
        try:
            parts = str(ip_str).split(".")
            if len(parts) != 4:
                return np.nan
            return float(int(parts[0]))
        except Exception:
            return np.nan

    df = df.copy()
    df["IsPrivateIP"] = df[col].map(_is_private)
    df["IpFirstOctet"] = df[col].map(_first_octet)

    df.drop(columns=[col], inplace=True)
    return df


def add_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived ratio features. Protected divisions to avoid inf.
    """
    df = df.copy()

    # MonetaryPerDay = MonetaryTotal / (Recency + 1)
    if {"MonetaryTotal", "Recency"}.issubset(df.columns):
        df["MonetaryPerDay"] = df["MonetaryTotal"] / (df["Recency"].astype(float) + 1.0)

    # AvgBasketValue = MonetaryTotal / Frequency
    if {"MonetaryTotal", "Frequency"}.issubset(df.columns):
        denom = df["Frequency"].replace(0, np.nan).astype(float)
        df["AvgBasketValue"] = df["MonetaryTotal"] / denom

    # TenureRatio = Recency / CustomerTenureDays
    if {"Recency", "CustomerTenureDays"}.issubset(df.columns):
        denom = df["CustomerTenureDays"].replace(0, np.nan).astype(float)
        df["TenureRatio"] = df["Recency"].astype(float) / denom

    return df


def drop_useless_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove columns known to be useless or problematic:
    - CustomerID (identifier)
    - Constant columns (variance nulle)
    """
    df = df.copy()

    if "CustomerID" in df.columns:
        df.drop(columns=["CustomerID"], inplace=True)

    nunique = df.nunique(dropna=False)
    constant_cols = nunique[nunique <= 1].index.tolist()
    if constant_cols:
        print(f"[Drop constants] Colonnes constantes supprimées : {constant_cols}")
        df.drop(columns=constant_cols, inplace=True)

    return df


# -----------------------------
# Column definitions (encoding)
# -----------------------------

def get_ordinal_mappings() -> Dict[str, List[str]]:
    """Ordinal categories per statement (ordered lists)."""
    return {
        "AgeCategory":        ["18-24", "25-34", "35-44", "45-54", "55-64", "65+", "Inconnu"],
        "SpendingCategory":   ["Low", "Medium", "High", "VIP"],
        "LoyaltyLevel":       ["Nouveau", "Jeune", "Établi", "Ancien", "Inconnu"],
        "ChurnRiskCategory":  ["Faible", "Moyen", "Élevé", "Critique"],
        "BasketSizeCategory": ["Petit", "Moyen", "Grand", "Inconnu"],
        "PreferredTimeOfDay": ["Matin", "Midi", "Après-midi", "Soir", "Nuit"],
    }


def infer_feature_groups(df: pd.DataFrame) -> Tuple[List[str], List[str], List[str]]:
    """Split columns into numeric / ordinal categorical / nominal categorical."""
    ordinal_map = get_ordinal_mappings()

    ordinal_cols = [c for c in ordinal_map.keys() if c in df.columns]

    numeric_cols = df.select_dtypes(
        include=["int64", "float64", "int32", "float32"]
    ).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != TARGET_COL]

    nominal_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()
    nominal_cols = [
        c for c in nominal_cols if c != TARGET_COL and c not in ordinal_cols
    ]

    for extra in ["RegYear", "RegMonth", "RegDay", "RegWeekday", "IsPrivateIP", "IpFirstOctet"]:
        if extra in df.columns and extra not in numeric_cols and extra != TARGET_COL:
            numeric_cols.append(extra)

    def _unique(seq: List[str]) -> List[str]:
        seen = set()
        out = []
        for x in seq:
            if x not in seen:
                seen.add(x)
                out.append(x)
        return out

    return _unique(numeric_cols), _unique(ordinal_cols), _unique(nominal_cols)


# -----------------------------
# Pipeline builder
# -----------------------------

def build_preprocessor(
    numeric_cols: List[str],
    ordinal_cols: List[str],
    nominal_cols: List[str],
) -> ColumnTransformer:
    ordinal_map = get_ordinal_mappings()

    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])

    ordinal_categories = [ordinal_map[c] for c in ordinal_cols]
    ordinal_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(
            categories=ordinal_categories,
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        )),
    ])

    nominal_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot",  OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("ord", ordinal_pipeline, ordinal_cols),
            ("nom", nominal_pipeline, nominal_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return preprocessor


def get_feature_names(preprocessor: ColumnTransformer) -> List[str]:
    """After fit, recover feature names for saving to CSV."""
    try:
        return preprocessor.get_feature_names_out().tolist()
    except Exception:
        return []


# ----------------------------------------
# Suppression par corrélation (NOUVEAU)
# ----------------------------------------

def remove_correlated_features(
        
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    threshold: float = 0.90,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Supprime les features numériques dont la corrélation de Pearson
    (en valeur absolue) avec une autre feature dépasse le seuil.

    Stratégie : pour chaque paire corrélée, on supprime la seconde feature
    (celle qui apparaît en colonne dans la matrice triangulaire supérieure).
    Le calcul est effectué uniquement sur X_train pour éviter le data leakage.
    La même liste de colonnes est ensuite supprimée de X_test.

    Parameters
    ----------
    X_train    : DataFrame transformé (train)
    X_test     : DataFrame transformé (test)
    threshold  : seuil de corrélation absolue (défaut : 0.90)

    Returns
    -------
    X_train_out, X_test_out, liste des colonnes supprimées
    """
    corr_matrix = X_train.corr(numeric_only=True).abs()

    # Masque triangulaire supérieure (sans la diagonale)
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )

    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]

    X_train_out = X_train.drop(columns=to_drop, errors="ignore")
    X_test_out  = X_test.drop(columns=to_drop, errors="ignore")

    if to_drop:
        print(f"[Corrélation] {len(to_drop)} feature(s) supprimée(s) "
              f"(seuil={threshold}) : {to_drop}")
    else:
        print(f"[Corrélation] Aucune feature supprimée (seuil={threshold}).")

    return X_train_out, X_test_out, to_drop


# ----------------------------------------
# Suppression des features corrélées à la cible (détection de leakage)
# ----------------------------------------

def remove_leakage_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    leakage_threshold: float = 0.3,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Supprime les features numériques qui sont trop corrélées avec la cible.
    Cela peut indiquer une fuite de données ou une feature trop directement liée au label.

    Parameters
    ----------
    X_train           : DataFrame transformé (train)
    X_test            : DataFrame transformé (test)
    y_train           : Labels d'entraînement
    leakage_threshold : seuil de corrélation absolue avec la cible (défaut : 0.3)

    Returns
    -------
    X_train_out, X_test_out, liste des colonnes supprimées
    """
    numeric_cols = X_train.select_dtypes(
        include=["int64", "float64", "int32", "float32"]
    ).columns.tolist()

    if not numeric_cols or len(y_train) == 0:
        return X_train, X_test, []

    to_drop = []
    for col in numeric_cols:
        try:
            corr = X_train[[col]].corrwith(y_train).iloc[0]
            if not np.isnan(corr) and abs(corr) > leakage_threshold:
                to_drop.append(col)
        except Exception:
            pass

    X_train_out = X_train.drop(columns=to_drop, errors="ignore")
    X_test_out = X_test.drop(columns=to_drop, errors="ignore")

    if to_drop:
        print(
            f"[Leakage] {len(to_drop)} feature(s) supprimée(s) "
            f"(corrélation cible > {leakage_threshold}) : {to_drop}"
        )
    else:
        print(f"[Leakage] Aucune fuite détectée (seuil={leakage_threshold}).")

    return X_train_out, X_test_out, to_drop


# ----------------------------------------
# Suppression par VIF (NOUVEAU)
# ----------------------------------------

def remove_high_vif_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    vif_threshold: float = 10.0,
    max_columns: int = 25,
    max_iterations: int = 20,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Suppression itérative des features avec un VIF (Variance Inflation Factor)
    trop élevé, indicateur de multicolinéarité.

    À chaque itération :
      1. Calcule le VIF de toutes les colonnes numériques sur X_train.
      2. Supprime la feature avec le VIF le plus élevé si elle dépasse le seuil.
      3. Répète jusqu'à ce que toutes les features soient sous le seuil.

    Le calcul est effectué uniquement sur X_train (pas de leakage).
    La même liste de colonnes est ensuite supprimée de X_test.

    Requiert : pip install statsmodels

    Parameters
    ----------
    X_train       : DataFrame transformé (train)
    X_test        : DataFrame transformé (test)
    vif_threshold : seuil VIF (défaut : 10.0)

    Returns
    -------
    X_train_out, X_test_out, liste des colonnes supprimées
    """
    try:
        from statsmodels.stats.outliers_influence import variance_inflation_factor
    except ImportError:
        print(
            "[VIF] statsmodels non installé. Étape VIF ignorée. "
            "Installez avec : pip install statsmodels"
        )
        return X_train, X_test, []

    num_cols = X_train.select_dtypes(
        include=["int64", "float64", "int32", "float32"]
    ).columns.tolist()

    if not num_cols:
        print("[VIF] Aucune colonne numérique détectée. Étape VIF ignorée.")
        return X_train, X_test, []

    if len(num_cols) > max_columns:
        print(
            f"[VIF] {len(num_cols)} colonnes numériques détectées. "
            f"Étape VIF ignorée au-delà de {max_columns} colonnes pour garder un temps d'exécution raisonnable."
        )
        return X_train, X_test, []

    dropped: List[str] = []
    current_cols = num_cols.copy()
    iterations = 0

    while True:
        iterations += 1
        if iterations > max_iterations:
            print(
                f"[VIF] Arrêt anticipé après {max_iterations} itérations pour limiter le temps de calcul."
            )
            break

        X_vif = X_train[current_cols].dropna()

        if X_vif.shape[1] < 2:
            break

        vif_data = pd.Series(
            [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])],
            index=current_cols,
        )

        max_vif = vif_data.max()
        if max_vif <= vif_threshold:
            break

        worst_col = vif_data.idxmax()
        print(f"[VIF] Suppression de '{worst_col}' (VIF={max_vif:.2f} > {vif_threshold})")
        current_cols.remove(worst_col)
        dropped.append(worst_col)

    if not dropped:
        print(f"[VIF] Aucune feature supprimée (seuil={vif_threshold}).")

    X_train_out = X_train.drop(columns=dropped, errors="ignore")
    X_test_out  = X_test.drop(columns=dropped, errors="ignore")

    return X_train_out, X_test_out, dropped


# -----------------------------
# Main preprocessing routine
# -----------------------------

def load_raw_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Raw dataset not found at: {path}")
    return pd.read_csv(path)


def prepare_dataframe(df: pd.DataFrame, require_target: bool = True) -> pd.DataFrame:
    """
    Opérations déterministes sans fit (pas de leakage) :
    - parsing dates → nouvelles colonnes
    - featurisation IP
    - ratios dérivés
    - suppression colonnes inutiles (ID, constantes)
    """
    df = df.copy()

    if require_target and TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found in dataset.")

    df = parse_registration_date(df, "RegistrationDate")
    df = featurize_ip(df, "LastLoginIP")
    df = add_feature_engineering(df)
    df = drop_useless_features(df)

    return df


def split_and_transform(
    df: pd.DataFrame,
    output_dir: Path,
    processed_path: Optional[Path] = None,
    test_size: float = 0.2,
    random_state: int = RANDOM_STATE,
    corr_threshold: float = 0.90,
    vif_threshold: float = 10.0,
    enable_vif: bool = False,
    leakage_threshold: float = 0.3,
) -> None:
    """
    Pipeline complet :
      1. Split stratifié train/test
      2. Inférence des groupes de features
      3. Fit du préprocesseur sklearn sur train uniquement
      4. Transformation train et test
      5. Suppression par corrélation de Pearson  ← NOUVEAU
      6. Suppression par VIF (multicolinéarité)  ← optionnelle
      7. Suppression des features corrélées à la cible (détection leakage) ← NOUVEAU
      8. Sauvegarde des CSVs

    Parameters
    ----------
    corr_threshold : seuil corrélation absolue (défaut 0.90)
    vif_threshold  : seuil VIF (défaut 10.0)
    leakage_threshold : seuil corrélation cible (défaut 0.3)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    if processed_path is not None:
        processed_path.parent.mkdir(parents=True, exist_ok=True)

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL].astype(int)

    # Étape 1 : split stratifié
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    # Étape 2 : inférence des groupes
    numeric_cols, ordinal_cols, nominal_cols = infer_feature_groups(X_train)

    # Étape 3 : build + fit du préprocesseur
    preprocessor = build_preprocessor(numeric_cols, ordinal_cols, nominal_cols)

    # Étape 4 : transformation (fit uniquement sur train)
    X_train_arr = preprocessor.fit_transform(X_train)
    X_test_arr  = preprocessor.transform(X_test)

    feature_names = get_feature_names(preprocessor)
    if feature_names and len(feature_names) == X_train_arr.shape[1]:
        X_train_df = pd.DataFrame(X_train_arr, columns=feature_names)
        X_test_df  = pd.DataFrame(X_test_arr,  columns=feature_names)
    else:
        X_train_df = pd.DataFrame(X_train_arr)
        X_test_df  = pd.DataFrame(X_test_arr)

    print(f"\n[Préprocesseur] Features après encodage : {X_train_df.shape[1]}")

    # Étape 5 : suppression par corrélation (NOUVEAU)
    X_train_df, X_test_df, dropped_corr = remove_correlated_features(
        X_train_df, X_test_df, threshold=corr_threshold
    )

    # Étape 6 : suppression par VIF (optionnelle car coûteuse)
    if enable_vif:
        X_train_df, X_test_df, dropped_vif = remove_high_vif_features(
            X_train_df,
            X_test_df,
            vif_threshold=vif_threshold,
        )
    else:
        dropped_vif = []
        print("[VIF] Étape désactivée par défaut pour réduire le temps d'exécution.")

    # Étape 7 : suppression des features corrélées à la cible (détection leakage)
    X_train_df, X_test_df, dropped_leakage = remove_leakage_features(
        X_train_df,
        X_test_df,
        y_train,
        leakage_threshold=leakage_threshold,
    )

    print(
        f"\n[Résumé] Features finales : {X_train_df.shape[1]} "
        f"| Supprimées corrélation : {len(dropped_corr)} "
        f"| Supprimées VIF : {len(dropped_vif)}"
    )

    # Étape 7 : sauvegarde
    X_train_df.to_csv(output_dir / "X_train.csv", index=False)
    X_test_df.to_csv(output_dir  / "X_test.csv",  index=False)
    y_train.to_frame(name=TARGET_COL).to_csv(output_dir / "y_train.csv", index=False)
    y_test.to_frame(name=TARGET_COL).to_csv(output_dir  / "y_test.csv",  index=False)

    # Sauvegarde optionnelle du dataset complet transformé
    if processed_path is not None:
        X_all_arr = preprocessor.transform(X)
        if feature_names and len(feature_names) == X_all_arr.shape[1]:
            X_all_df = pd.DataFrame(X_all_arr, columns=feature_names)
        else:
            X_all_df = pd.DataFrame(X_all_arr)

        # Appliquer exactement les mêmes suppressions que sur train
        cols_to_keep = X_train_df.columns.tolist()
        X_all_df = X_all_df[[c for c in cols_to_keep if c in X_all_df.columns]]

        df_processed = X_all_df.copy()
        df_processed[TARGET_COL] = y.values
        df_processed.to_csv(processed_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Retail churn preprocessing pipeline")
    parser.add_argument("--raw-path",        type=str,   default=str(DEFAULT_RAW_PATH))
    parser.add_argument("--output-dir",      type=str,   default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--processed-path",  type=str,   default=str(DEFAULT_PROCESSED_PATH))
    parser.add_argument("--no-processed-save", action="store_true",
                        help="Ne pas sauvegarder le dataset complet transformé")
    parser.add_argument("--test-size",       type=float, default=0.2)
    parser.add_argument("--corr-threshold",  type=float, default=0.90,
                        help="Seuil de corrélation de Pearson absolue (défaut: 0.90)")
    parser.add_argument("--vif-threshold",   type=float, default=10.0,
                        help="Seuil VIF pour la multicolinéarité (défaut: 10.0)")
    parser.add_argument("--enable-vif", action="store_true",
                        help="Activer la suppression VIF, plus lente mais optionnelle")
    parser.add_argument("--leakage-threshold", type=float, default=0.3,
                        help="Seuil corrélation cible pour détecter les fuites (défaut: 0.3)")
    args = parser.parse_args()

    raw_path       = Path(args.raw_path)
    output_dir     = Path(args.output_dir)
    processed_path = None if args.no_processed_save else Path(args.processed_path)

    df_raw      = load_raw_dataset(raw_path)
    df_prepared = prepare_dataframe(df_raw)

    split_and_transform(
        df=df_prepared,
        output_dir=output_dir,
        processed_path=processed_path,
        test_size=args.test_size,
        random_state=RANDOM_STATE,
        corr_threshold=args.corr_threshold,
        vif_threshold=args.vif_threshold,
        enable_vif=args.enable_vif,
        leakage_threshold=args.leakage_threshold,
    )

    print("\nPrétraitement terminé.")
    print(f"Train/test sauvegardés dans : {output_dir}")
    if processed_path is not None:
        print(f"Dataset complet transformé sauvegardé dans : {processed_path}")


if __name__ == "__main__":
    main()
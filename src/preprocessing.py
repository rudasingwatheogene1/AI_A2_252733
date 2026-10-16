from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

SEED = 7330

TARGET_REGRESSION = "yield_tons"
TARGET_CLASSIFICATION = "high_risk"

REGRESSION_TARGET = TARGET_REGRESSION
CLASSIFICATION_TARGET = TARGET_CLASSIFICATION

NUMERIC_FEATURES = [
    "rainfall_mm",
    "avg_temp_c",
    "soil_moisture_pct",
    "fertilizer_kg",
    "field_size_ha",
    "cooperative_support",
]

CATEGORICAL_FEATURES = [
    "season",
]

REGRESSION_TARGET = "yield_tons"
CLASSIFICATION_TARGET = "high_risk"


# Important:
# These targets and derived target information must NOT
# be used as classification input features.
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def create_preprocessor() -> ColumnTransformer:
    """
    Create a preprocessing pipeline.

    Numeric:
        median imputation + standard scaling

    Categorical:
        most-frequent imputation + one-hot encoding
    """

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                NUMERIC_FEATURES,
            ),
            (
                "categorical",
                categorical_pipeline,
                CATEGORICAL_FEATURES,
            ),
        ]
    )

    return preprocessor


def prepare_data(
    df: pd.DataFrame,
    target: str,
):
    """
    Split the dataset into training and testing sets.

    The split happens BEFORE preprocessing so that
    transformations are fitted only on training data.
    """

    X = df[FEATURES].copy()
    y = df[target].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=SEED,
        stratify=y if target == TARGET_CLASSIFICATION else None,
    )

    return (
        X_train,
        X_test,
        y_train,
        y_test,
    )


def save_preprocessor(
    preprocessor,
    filename: str = "preprocessor.joblib",
):
    """Save a fitted preprocessing object."""

    models_dir = Path("models")
    models_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = models_dir / filename

    joblib.dump(
        preprocessor,
        path,
    )

    return path
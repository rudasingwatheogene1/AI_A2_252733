from pathlib import Path
import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.pipeline import Pipeline

from src.preprocessing import (
    create_preprocessor,
    prepare_data,
)


SEED = 7330

DATASET = Path("data/AI_A2_2527330.csv")

METRICS_FILE = Path(
    "artifacts/regression_metrics.json"
)

PREDICTIONS_FILE = Path(
    "artifacts/test_predictions.csv"
)

RESIDUAL_PLOT = Path(
    "artifacts/plots/residuals.png"
)

MODEL_FILE = Path(
    "models/regression_model.joblib"
)

VERIFICATION_CODE = "AI-A2-7330"


def load_data() -> pd.DataFrame:
    """Load the generated dataset."""

    if not DATASET.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET}"
        )

    return pd.read_csv(DATASET)


def calculate_metrics(
    y_true,
    y_pred,
) -> dict:
    """Calculate required regression metrics."""

    mae = mean_absolute_error(
        y_true,
        y_pred,
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            y_pred,
        )
    )

    r2 = r2_score(
        y_true,
        y_pred,
    )

    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "R2": float(r2),
    }


def create_models():
    """Create candidate regression models."""

    linear_pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                create_preprocessor(),
            ),
            (
                "model",
                LinearRegression(),
            ),
        ]
    )

    random_forest_pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                create_preprocessor(),
            ),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=300,
                    random_state=SEED,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    return {
        "Linear Regression": linear_pipeline,
        "Random Forest": random_forest_pipeline,
    }


def train_regression():
    """Train and evaluate regression models."""

    df = load_data()

    (
        X_train,
        X_test,
        y_train,
        y_test,
    ) = prepare_data(
        df,
        target="yield_tons",
    )

    # --------------------------------------------------
    # Preserve original test-set row IDs
    # --------------------------------------------------

    test_row_ids = y_test.index.to_numpy()

    # --------------------------------------------------
    # Mean baseline
    # --------------------------------------------------

    baseline_prediction = np.full(
        len(y_test),
        y_train.mean(),
    )

    baseline_metrics = calculate_metrics(
        y_test,
        baseline_prediction,
    )

    # --------------------------------------------------
    # Candidate models
    # --------------------------------------------------

    models = create_models()

    results = {
        "student_id": "25/27330",
        "verification_code": VERIFICATION_CODE,
        "seed": SEED,
        "test_size": 0.20,
        "baseline": {
            "name": "Mean Baseline",
            "metrics": baseline_metrics,
        },
        "candidates": {},
    }

    # IMPORTANT:
    # row_id identifies the original observation.
    predictions = pd.DataFrame(
        {
            "row_id": test_row_ids,
            "actual_yield_tons": y_test.to_numpy(),
        }
    )

    best_name = None
    best_model = None
    best_r2 = -np.inf
    best_prediction = None

    for name, model in models.items():

        print(f"\nTraining: {name}")

        model.fit(
            X_train,
            y_train,
        )

        prediction = model.predict(
            X_test
        )

        metrics = calculate_metrics(
            y_test,
            prediction,
        )

        results["candidates"][name] = {
            "metrics": metrics
        }

        column_name = (
            "predicted_"
            + name.lower().replace(" ", "_")
        )

        predictions[column_name] = prediction

        print(
            f"MAE:  {metrics['MAE']:.4f}"
        )

        print(
            f"RMSE: {metrics['RMSE']:.4f}"
        )

        print(
            f"R2:   {metrics['R2']:.4f}"
        )

        # Select model using highest R².
        if metrics["R2"] > best_r2:
            best_r2 = metrics["R2"]
            best_name = name
            best_model = model
            best_prediction = prediction

    # --------------------------------------------------
    # Save selected model
    # --------------------------------------------------

    results["selected_model"] = best_name

    results["selection_reason"] = (
        "Selected the candidate with the highest "
        "held-out test R2."
    )

    MODEL_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        best_model,
        MODEL_FILE,
    )

    # --------------------------------------------------
    # Save test predictions
    # --------------------------------------------------

    predictions["selected_model"] = best_name

    predictions["selected_prediction"] = (
        best_prediction
    )

    predictions["residual"] = (
        predictions["actual_yield_tons"]
        - predictions["selected_prediction"]
    )

    PREDICTIONS_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    predictions.to_csv(
        PREDICTIONS_FILE,
        index=False,
    )

    # --------------------------------------------------
    # Residual plot
    # --------------------------------------------------

    RESIDUAL_PLOT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    residuals = (
        y_test.to_numpy()
        - best_prediction
    )

    plt.figure(
        figsize=(8, 5)
    )

    plt.scatter(
        best_prediction,
        residuals,
    )

    plt.axhline(
        0,
        linestyle="--",
    )

    plt.xlabel(
        "Predicted Yield (tons)"
    )

    plt.ylabel(
        "Residual"
    )

    plt.title(
        f"Residual Plot - {best_name}"
    )

    plt.tight_layout()

    plt.savefig(
        RESIDUAL_PLOT,
        dpi=150,
    )

    plt.close()

    # --------------------------------------------------
    # Save metrics
    # --------------------------------------------------

    METRICS_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        METRICS_FILE,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            results,
            file,
            indent=4,
        )

    # --------------------------------------------------
    # Final output
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("REGRESSION RESULTS")
    print("=" * 60)

    print(
        f"Selected model: {best_name}"
    )

    print(
        f"Best test R2: {best_r2:.4f}"
    )

    print(
        f"Model saved: {MODEL_FILE}"
    )

    print(
        f"Metrics saved: {METRICS_FILE}"
    )

    print(
        f"Predictions saved: {PREDICTIONS_FILE}"
    )

    print(
        f"Plot saved: {RESIDUAL_PLOT}"
    )

    print("=" * 60)

    return results


if __name__ == "__main__":
    train_regression()
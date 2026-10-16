import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.pipeline import Pipeline

from src.preprocessing import (
    create_preprocessor,
    prepare_data,
    FEATURES,
    CLASSIFICATION_TARGET,
    SEED,
)


BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "AI_A2_2527330.csv"
MODEL_PATH = BASE_DIR / "models" / "classification_model.joblib"
METRICS_PATH = BASE_DIR / "artifacts" / "classification_metrics.json"
CM_PATH = BASE_DIR / "artifacts" / "plots" / "confusion_matrix.png"
PREDICTIONS_PATH = BASE_DIR / "artifacts" / "test_predictions.csv"


VERIFICATION_CODE = "AI-A2-7330"


def main():

    print("=" * 60)
    print("MUSANZE HARVEST INTELLIGENCE SERVICE")
    print("CLASSIFICATION")
    print("=" * 60)
    print("Student ID:        25/27330")
    print("Verification Code: AI-A2-7330")
    print("Seed:              7330")
    print("=" * 60)

    # ---------------------------------------------------------
    # Load data
    # ---------------------------------------------------------

    df = pd.read_csv(DATA_PATH)

    # Preserve original dataset row ID.
    # This allows regression and classification evidence
    # to be joined using the actual observation.
    df["_row_id"] = df.index

    X_train, X_test, y_train, y_test = prepare_data(
        df.drop(columns=["_row_id"]),
        target=CLASSIFICATION_TARGET,
    )

    # Recover the original row IDs using the indexes returned
    # by train_test_split.
    test_row_ids = y_test.index.to_numpy()

    # ---------------------------------------------------------
    # Majority-class baseline
    # ---------------------------------------------------------

    majority_class = y_train.mode()[0]

    baseline_pred = [majority_class] * len(y_test)

    baseline_f1 = f1_score(
        y_test,
        baseline_pred,
        zero_division=0,
    )

    baseline_accuracy = accuracy_score(
        y_test,
        baseline_pred,
    )

    print("\nMajority Class Baseline")
    print(f"Accuracy: {baseline_accuracy:.4f}")
    print(f"F1:       {baseline_f1:.4f}")

    # ---------------------------------------------------------
    # Candidate 1: Logistic Regression
    # ---------------------------------------------------------

    print("\nTraining: Logistic Regression")

    logistic_model = Pipeline(
        steps=[
            ("preprocessor", create_preprocessor()),
            (
                "model",
                LogisticRegression(
                    max_iter=2000,
                    random_state=SEED,
                ),
            ),
        ]
    )

    logistic_model.fit(X_train, y_train)

    logistic_pred = logistic_model.predict(X_test)
    logistic_prob = logistic_model.predict_proba(X_test)[:, 1]

    logistic_metrics = {
        "accuracy": float(
            accuracy_score(y_test, logistic_pred)
        ),
        "precision": float(
            precision_score(
                y_test,
                logistic_pred,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_test,
                logistic_pred,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                y_test,
                logistic_pred,
                zero_division=0,
            )
        ),
        "roc_auc": float(
            roc_auc_score(
                y_test,
                logistic_prob,
            )
        ),
    }

    print(f"Accuracy:  {logistic_metrics['accuracy']:.4f}")
    print(f"Precision: {logistic_metrics['precision']:.4f}")
    print(f"Recall:    {logistic_metrics['recall']:.4f}")
    print(f"F1:        {logistic_metrics['f1']:.4f}")
    print(f"ROC-AUC:   {logistic_metrics['roc_auc']:.4f}")

    # ---------------------------------------------------------
    # Candidate 2: Random Forest
    # ---------------------------------------------------------

    print("\nTraining: Random Forest")

    rf_model = Pipeline(
        steps=[
            ("preprocessor", create_preprocessor()),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=300,
                    random_state=SEED,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    rf_model.fit(X_train, y_train)

    rf_pred = rf_model.predict(X_test)
    rf_prob = rf_model.predict_proba(X_test)[:, 1]

    rf_metrics = {
        "accuracy": float(
            accuracy_score(y_test, rf_pred)
        ),
        "precision": float(
            precision_score(
                y_test,
                rf_pred,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_test,
                rf_pred,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                y_test,
                rf_pred,
                zero_division=0,
            )
        ),
        "roc_auc": float(
            roc_auc_score(
                y_test,
                rf_prob,
            )
        ),
    }

    print(f"Accuracy:  {rf_metrics['accuracy']:.4f}")
    print(f"Precision: {rf_metrics['precision']:.4f}")
    print(f"Recall:    {rf_metrics['recall']:.4f}")
    print(f"F1:        {rf_metrics['f1']:.4f}")
    print(f"ROC-AUC:   {rf_metrics['roc_auc']:.4f}")

    # ---------------------------------------------------------
    # Select model using F1
    # ---------------------------------------------------------

    candidates = {
        "Logistic Regression": (
            logistic_model,
            logistic_metrics,
            logistic_pred,
            logistic_prob,
        ),
        "Random Forest": (
            rf_model,
            rf_metrics,
            rf_pred,
            rf_prob,
        ),
    }

    selected_name = max(
        candidates,
        key=lambda name: candidates[name][1]["f1"],
    )

    (
        selected_model,
        selected_metrics,
        selected_pred,
        selected_prob,
    ) = candidates[selected_name]

    # ---------------------------------------------------------
    # Save model
    # ---------------------------------------------------------

    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        selected_model,
        MODEL_PATH,
    )

    # ---------------------------------------------------------
    # Save metrics
    # ---------------------------------------------------------

    METRICS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results = {
        "student_id": "25/27330",
        "verification_code": VERIFICATION_CODE,
        "seed": SEED,
        "target": CLASSIFICATION_TARGET,
        "baseline": {
            "name": "Majority Class",
            "accuracy": float(baseline_accuracy),
            "f1": float(baseline_f1),
        },
        "candidates": {
            "Logistic Regression": logistic_metrics,
            "Random Forest": rf_metrics,
        },
        "selected_model": selected_name,
        "selection_metric": "f1",
        "selected_metrics": selected_metrics,
    }

    with open(
        METRICS_PATH,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            results,
            f,
            indent=4,
        )

    # ---------------------------------------------------------
    # Confusion matrix
    # ---------------------------------------------------------

    cm = confusion_matrix(
        y_test,
        selected_pred,
    )

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=[
            "Low Risk",
            "High Risk",
        ],
    )

    fig, ax = plt.subplots(
        figsize=(6, 5)
    )

    disp.plot(
        ax=ax
    )

    ax.set_title(
        f"Classification Confusion Matrix\n"
        f"AI-A2-7330 | {selected_name}"
    )

    CM_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.tight_layout()

    fig.savefig(
        CM_PATH,
        dpi=150,
    )

    plt.close(fig)

    # ---------------------------------------------------------
    # Create classification evidence
    # ---------------------------------------------------------

    classification_predictions = pd.DataFrame(
        {
            "row_id": test_row_ids,
            "classification_actual": y_test.to_numpy(),
            "classification_predicted": selected_pred,
            "high_risk_probability": selected_prob,
        }
    )

    # ---------------------------------------------------------
    # Merge with regression evidence using row_id
    # ---------------------------------------------------------

    if PREDICTIONS_PATH.exists():

        regression_predictions = pd.read_csv(
            PREDICTIONS_PATH
        )

        # Remove pandas' old unnamed index column if present.
        regression_predictions = regression_predictions.loc[
            :,
            ~regression_predictions.columns.str.startswith(
                "Unnamed:"
            ),
        ]

        # Regression script should provide row_id.
        if "row_id" in regression_predictions.columns:

            combined = regression_predictions.merge(
                classification_predictions,
                on="row_id",
                how="outer",
                validate="one_to_one",
            )

        else:

            # If an old artifact exists without row_id,
            # create a clean classification-only evidence file.
            combined = classification_predictions.copy()

    else:

        combined = classification_predictions.copy()

    combined.to_csv(
        PREDICTIONS_PATH,
        index=False,
    )

    # ---------------------------------------------------------
    # Final output
    # ---------------------------------------------------------

    print("\n" + "=" * 60)
    print("CLASSIFICATION RESULTS")
    print("=" * 60)

    print(
        f"Selected model: {selected_name}"
    )

    print(
        f"Best F1:        "
        f"{selected_metrics['f1']:.4f}"
    )

    print(
        f"ROC-AUC:        "
        f"{selected_metrics['roc_auc']:.4f}"
    )

    print(
        f"Model saved:    {MODEL_PATH}"
    )

    print(
        f"Metrics saved:  {METRICS_PATH}"
    )

    print(
        f"Plot saved:     {CM_PATH}"
    )

    print(
        f"Predictions:    {PREDICTIONS_PATH}"
    )

    print("=" * 60)


if __name__ == "__main__":
    main()
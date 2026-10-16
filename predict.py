import json
from pathlib import Path

import joblib
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

VERIFICATION_CODE = "AI-A2-7330"

regression_model = joblib.load(
    MODEL_DIR / "regression_model.joblib"
)

classification_model = joblib.load(
    MODEL_DIR / "classification_model.joblib"
)

clustering_model = joblib.load(
    MODEL_DIR / "clustering_model.joblib"
)


FEATURES = [
    "rainfall_mm",
    "avg_temp_c",
    "soil_moisture_pct",
    "fertilizer_kg",
    "field_size_ha",
    "season",
    "cooperative_support",
]

CLUSTER_FEATURES = [
    "rainfall_mm",
    "avg_temp_c",
    "soil_moisture_pct",
    "fertilizer_kg",
    "field_size_ha",
    "cooperative_support",
]


def validate_input(data: dict):
    missing = [field for field in FEATURES if field not in data]

    if missing:
        raise ValueError(
            f"Missing required fields: {', '.join(missing)}"
        )

    numeric_fields = [
        "rainfall_mm",
        "avg_temp_c",
        "soil_moisture_pct",
        "fertilizer_kg",
        "field_size_ha",
    ]

    for field in numeric_fields:
        if not isinstance(data[field], (int, float)):
            raise TypeError(f"{field} must be numeric.")

    ranges = {
        "rainfall_mm": (600, 1400),
        "avg_temp_c": (12, 26),
        "soil_moisture_pct": (20, 80),
        "fertilizer_kg": (0, 250),
        "field_size_ha": (0.2, 4.0),
    }

    for field, (minimum, maximum) in ranges.items():
        value = data[field]

        if value < minimum or value > maximum:
            raise ValueError(
                f"{field} must be between {minimum} and {maximum}."
            )

    if data["season"] not in ["A", "B"]:
        raise ValueError("season must be either 'A' or 'B'.")

    if data["cooperative_support"] not in [0, 1]:
        raise ValueError(
            "cooperative_support must be either 0 or 1."
        )


def predict(data: dict):

    validate_input(data)

    X = pd.DataFrame(
        [[data[field] for field in FEATURES]],
        columns=FEATURES,
    )

    yield_prediction = float(
        regression_model.predict(X)[0]
    )

    risk_prediction = int(
        classification_model.predict(X)[0]
    )

    if hasattr(classification_model, "predict_proba"):

        probabilities = classification_model.predict_proba(X)[0]
        classes = classification_model.classes_

        if 1 in classes:
            risk_index = list(classes).index(1)
            risk_probability = float(
                probabilities[risk_index]
            )
        else:
            risk_probability = 0.0

    else:
        risk_probability = None

    X_cluster = X[CLUSTER_FEATURES]

    cluster = int(
        clustering_model.predict(X_cluster)[0]
    )

    return {
        "estimated_yield_tons": round(
            yield_prediction, 4
        ),
        "high_risk_prediction": risk_prediction,
        "high_risk_probability": (
            round(risk_probability, 4)
            if risk_probability is not None
            else None
        ),
        "cluster": cluster,
        "verification_code": VERIFICATION_CODE,
    }


def main():

    sample = {
        "rainfall_mm": 900,
        "avg_temp_c": 20,
        "soil_moisture_pct": 55,
        "fertilizer_kg": 100,
        "field_size_ha": 2.0,
        "season": "A",
        "cooperative_support": 1,
    }

    print("=" * 60)
    print("MUSANZE HARVEST INTELLIGENCE SERVICE")
    print("=" * 60)
    print("Student:       RUDASINGWA Theogene")
    print("Registration:  25/27330")
    print("Verification:  AI-A2-7330")
    print("-" * 60)

    try:

        result = predict(sample)

        print("Prediction successful!")
        print()
        print(json.dumps(result, indent=4))

    except Exception as error:

        print("Prediction failed:")
        print(error)


if __name__ == "__main__":
    main()

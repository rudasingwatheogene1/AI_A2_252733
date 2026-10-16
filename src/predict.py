import argparse
import json
import joblib
import pandas as pd
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent

STUDENT_NAME = "RUDASINGWA Theogene"
REGISTRATION_NUMBER = "25/27330"
VERIFICATION_CODE = "AI-A2-7330"

REQUIRED_FIELDS = [
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

RANGES = {
    "rainfall_mm": (600, 1400),
    "avg_temp_c": (12, 26),
    "soil_moisture_pct": (20, 80),
    "fertilizer_kg": (0, 250),
    "field_size_ha": (0.2, 4.0),
}


def load_models():
    regression_model = joblib.load(
        BASE_DIR / "models" / "regression_model.joblib"
    )

    classification_model = joblib.load(
        BASE_DIR / "models" / "classification_model.joblib"
    )

    clustering_model = joblib.load(
        BASE_DIR / "models" / "clustering_model.joblib"
    )

    return regression_model, classification_model, clustering_model


def validate_input(data):
    if not isinstance(data, dict):
        raise TypeError("Input must be a JSON object.")

    missing = [field for field in REQUIRED_FIELDS if field not in data]

    if missing:
        raise ValueError(
            f"Missing required field(s): {', '.join(missing)}"
        )

    for field in REQUIRED_FIELDS:
        if field not in ["season"]:
            if isinstance(data[field], bool):
                raise TypeError(
                    f"{field} must be numeric, not boolean."
                )

    for field, (minimum, maximum) in RANGES.items():
        value = data[field]

        if not isinstance(value, (int, float)):
            raise TypeError(f"{field} must be numeric.")

        if not minimum <= value <= maximum:
            raise ValueError(
                f"{field} must be between {minimum} and {maximum}."
            )

    if data["season"] not in ["A", "B"]:
        raise ValueError("season must be either 'A' or 'B'.")

    if data["cooperative_support"] not in [0, 1]:
        raise ValueError("cooperative_support must be 0 or 1.")


def predict(data):
    validate_input(data)

    regression_model, classification_model, clustering_model = load_models()

    row = pd.DataFrame([data])

    estimated_yield = regression_model.predict(row)[0]

    high_risk_prediction = int(
        classification_model.predict(row)[0]
    )

    probability = classification_model.predict_proba(row)[0][1]

    cluster_input = row[CLUSTER_FEATURES]

    cluster = int(
        clustering_model.predict(cluster_input)[0]
    )

    return {
        "estimated_yield_tons": round(float(estimated_yield), 4),
        "high_risk_prediction": high_risk_prediction,
        "high_risk_probability": round(float(probability), 4),
        "cluster": cluster,
        "verification_code": VERIFICATION_CODE,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Musanze Harvest Intelligence Service prediction CLI"
    )

    parser.add_argument(
        "--json",
        type=str,
        help="One JSON prediction record."
    )

    args = parser.parse_args()

    if args.json:
        try:
            data = json.loads(args.json)
            result = predict(data)

            print(json.dumps(result, indent=4))

        except json.JSONDecodeError as error:
            print(f"ERROR: Invalid JSON: {error}")
            raise SystemExit(1)

        except (ValueError, TypeError) as error:
            print(f"ERROR: {error}")
            raise SystemExit(1)

    else:
        sample = {
            "rainfall_mm": 1000,
            "avg_temp_c": 18,
            "soil_moisture_pct": 55,
            "fertilizer_kg": 120,
            "field_size_ha": 2.0,
            "season": "A",
            "cooperative_support": 1,
        }

        print(f"Student: {STUDENT_NAME}")
        print(f"Registration: {REGISTRATION_NUMBER}")
        print(f"Verification: {VERIFICATION_CODE}")
        print("\nPrediction successful!")

        result = predict(sample)

        print(json.dumps(result, indent=4))


if __name__ == "__main__":
    main()
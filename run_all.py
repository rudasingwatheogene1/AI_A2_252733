import argparse
import subprocess
import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent


def run_command(command, description):
    print()
    print("=" * 70)
    print(description)
    print("=" * 70)

    result = subprocess.run(
        command,
        cwd=BASE_DIR,
        text=True,
    )

    if result.returncode != 0:
        print()
        print(f"FAILED: {description}")
        sys.exit(result.returncode)


def main():

    parser = argparse.ArgumentParser(
        description="Musanze Harvest Intelligence Service - Reproducible ML Pipeline"
    )

    parser.add_argument(
        "--student-id",
        required=True,
        help="Student registration number, e.g. 25/27330",
    )

    parser.add_argument(
        "--seed",
        required=True,
        type=int,
        help="Last four registration digits, e.g. 7330",
    )

    args = parser.parse_args()

    expected_student_id = "25/27330"
    expected_seed = 7330

    print("=" * 70)
    print("MUSANZE HARVEST INTELLIGENCE SERVICE")
    print("FULL REPRODUCIBLE PIPELINE")
    print("=" * 70)

    print(f"Student ID: {args.student_id}")
    print(f"Seed:       {args.seed}")
    print("Expected:   25/27330 / 7330")
    print("=" * 70)

    # ---------------------------------------------------------
    # Check identity and seed
    # ---------------------------------------------------------

    if args.student_id != expected_student_id:
        print()
        print("ERROR: Incorrect student ID.")
        print(f"Expected: {expected_student_id}")
        print(f"Received: {args.student_id}")
        sys.exit(1)

    if args.seed != expected_seed:
        print()
        print("ERROR: Incorrect seed.")
        print(f"Expected: {expected_seed}")
        print(f"Received: {args.seed}")
        sys.exit(1)

    # ---------------------------------------------------------
    # Step 1: Generate personalized dataset
    # ---------------------------------------------------------

    run_command(
        [
            sys.executable,
            "data/generate_data.py",
        ],
        "STEP 1 - Generating personalized synthetic dataset",
    )

    # ---------------------------------------------------------
    # Step 2: Validate dataset
    # ---------------------------------------------------------

    run_command(
        [
            sys.executable,
            "validate_data.py",
        ],
        "STEP 2 - Validating dataset",
    )

    # ---------------------------------------------------------
    # Step 3: Regression
    # ---------------------------------------------------------

    run_command(
        [
            sys.executable,
            "-m",
            "src.regression",
        ],
        "STEP 3 - Training regression models",
    )

    # ---------------------------------------------------------
    # Step 4: Classification
    # ---------------------------------------------------------

    run_command(
        [
            sys.executable,
            "-m",
            "src.classification",
        ],
        "STEP 4 - Training classification models",
    )

    # ---------------------------------------------------------
    # Step 5: Clustering
    # ---------------------------------------------------------

    run_command(
        [
            sys.executable,
            "-m",
            "src.clustering",
        ],
        "STEP 5 - Training clustering models",
    )

    # ---------------------------------------------------------
    # Step 6: Prediction smoke test
    # ---------------------------------------------------------

    run_command(
        [
            sys.executable,
            "predict.py",
        ],
        "STEP 6 - Testing saved prediction models",
    )

    # ---------------------------------------------------------
    # Step 7: Automated tests
    # ---------------------------------------------------------

    run_command(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests",
            "-v",
        ],
        "STEP 7 - Running automated tests",
    )

    # ---------------------------------------------------------
    # Final artifact check
    # ---------------------------------------------------------

    required_files = [
        "data/AI_A2_2527330.csv",
        "artifacts/data_fingerprint.txt",
        "artifacts/regression_metrics.json",
        "artifacts/classification_metrics.json",
        "artifacts/clustering_report.json",
        "artifacts/test_predictions.csv",
        "artifacts/plots/residuals.png",
        "artifacts/plots/confusion_matrix.png",
        "artifacts/plots/clusters.png",
        "models/regression_model.joblib",
        "models/classification_model.joblib",
        "models/clustering_model.joblib",
        "predict.py",
        "app.py",
        "tests/test_system.py",
    ]

    print()
    print("=" * 70)
    print("FINAL ARTIFACT CHECK")
    print("=" * 70)

    missing = []

    for file_path in required_files:
        path = BASE_DIR / file_path

        if path.exists():
            print(f"[OK]      {file_path}")
        else:
            print(f"[MISSING] {file_path}")
            missing.append(file_path)

    print("=" * 70)

    if missing:
        print()
        print("PIPELINE FINISHED WITH MISSING ARTIFACTS.")
        sys.exit(1)

    print()
    print("=" * 70)
    print("SUCCESS - FULL PIPELINE COMPLETED")
    print("=" * 70)
    print("Student:       RUDASINGWA Theogene")
    print("Registration:  25/27330")
    print("Seed:          7330")
    print("Verification:  AI-A2-7330")
    print()
    print("Dataset generated")
    print("Data validated")
    print("Regression trained")
    print("Classification trained")
    print("Clustering trained")
    print("Prediction tested")
    print("Automated tests passed")
    print("Required artifacts checked")
    print()
    print("Musanze Harvest Intelligence Service is reproducible.")
    print("=" * 70)


if __name__ == "__main__":
    main()
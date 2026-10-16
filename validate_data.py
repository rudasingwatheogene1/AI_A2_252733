from src.validation import (
    validate_dataset,
    print_validation_report,
)


DATASET = "data/AI_A2_2527330.csv"


if __name__ == "__main__":
    report = validate_dataset(DATASET)
    print_validation_report(report)
from pathlib import Path
import pandas as pd


REQUIRED_COLUMNS = [
    "rainfall_mm",
    "avg_temp_c",
    "soil_moisture_pct",
    "fertilizer_kg",
    "field_size_ha",
    "season",
    "cooperative_support",
    "yield_tons",
    "high_risk",
]


NUMERIC_RANGES = {
    "rainfall_mm": (600, 1400),
    "avg_temp_c": (12, 26),
    "soil_moisture_pct": (20, 80),
    "fertilizer_kg": (0, 250),
    "field_size_ha": (0.2, 4.0),
}


def validate_columns(df: pd.DataFrame) -> list[str]:
    """Check that all required columns exist."""

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    return missing_columns


def validate_ranges(df: pd.DataFrame) -> dict:
    """Check numeric feature ranges."""

    results = {}

    for column, (minimum, maximum) in NUMERIC_RANGES.items():

        values = df[column].dropna()

        results[column] = {
            "minimum_allowed": minimum,
            "maximum_allowed": maximum,
            "actual_minimum": float(values.min()),
            "actual_maximum": float(values.max()),
            "valid": bool(
                (values >= minimum).all()
                and (values <= maximum).all()
            ),
        }

    return results


def validate_missing_values(df: pd.DataFrame) -> dict:
    """Return missing-value counts."""

    return df.isna().sum().to_dict()


def validate_duplicates(df: pd.DataFrame) -> int:
    """Return the number of duplicate rows."""

    return int(df.duplicated().sum())


def validate_targets(df: pd.DataFrame) -> dict:
    """Validate target variables."""

    return {
        "yield_tons_min": float(df["yield_tons"].min()),
        "yield_tons_valid": bool(
            (df["yield_tons"] >= 0.50).all()
        ),
        "high_risk_values": sorted(
            df["high_risk"].dropna().unique().tolist()
        ),
        "high_risk_valid": bool(
            set(df["high_risk"].dropna().unique()).issubset({0, 1})
        ),
    }


def validate_dataset(path: str | Path) -> dict:
    """Run all validation checks."""

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}"
        )

    df = pd.read_csv(path)

    missing_columns = validate_columns(df)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    report = {
        "rows": len(df),
        "columns": len(df.columns),
        "required_columns": REQUIRED_COLUMNS,
        "missing_values": validate_missing_values(df),
        "duplicate_rows": validate_duplicates(df),
        "ranges": validate_ranges(df),
        "targets": validate_targets(df),
    }

    return report


def print_validation_report(report: dict) -> None:
    """Print a readable validation report."""

    print("\n" + "=" * 60)
    print("DATA VALIDATION REPORT")
    print("=" * 60)

    print(f"Rows:              {report['rows']}")
    print(f"Columns:           {report['columns']}")
    print(f"Duplicate rows:    {report['duplicate_rows']}")

    print("\nMissing values:")

    for column, count in report["missing_values"].items():
        if count > 0:
            print(f"  {column}: {count}")

    print("\nRange checks:")

    for column, result in report["ranges"].items():
        print(
            f"  {column}: "
            f"{result['actual_minimum']:.4f} - "
            f"{result['actual_maximum']:.4f} "
            f"-> {result['valid']}"
        )

    print("\nTarget checks:")

    targets = report["targets"]

    print(
        f"  Minimum yield: {targets['yield_tons_min']:.4f}"
    )

    print(
        f"  Yield target valid: {targets['yield_tons_valid']}"
    )

    print(
        f"  High-risk values: {targets['high_risk_values']}"
    )

    print(
        f"  High-risk target valid: "
        f"{targets['high_risk_valid']}"
    )

    print("=" * 60)
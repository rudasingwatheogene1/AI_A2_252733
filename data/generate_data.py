from pathlib import Path
import hashlib
import numpy as np
import pandas as pd


STUDENT_ID = "25/27330"
SEED = 7330
VERIFICATION_CODE = "AI-A2-7330"

BASE_ROWS = 240 + (SEED % 41)

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
ARTIFACTS_DIR = ROOT_DIR / "artifacts"

DATA_FILE = DATA_DIR / "AI_A2_2527330.csv"
FINGERPRINT_FILE = ARTIFACTS_DIR / "data_fingerprint.txt"


def generate_dataset(seed: int = SEED) -> pd.DataFrame:
    """
    Generate the personalized synthetic Musanze potato dataset.
    """

    if seed != SEED:
        raise ValueError(
            f"Invalid seed. This submission requires seed {SEED}."
        )

    rng = np.random.default_rng(seed)

    n = BASE_ROWS

    rainfall = rng.uniform(600, 1400, n)
    temperature = rng.uniform(12, 26, n)
    soil_moisture = rng.uniform(20, 80, n)
    fertilizer = rng.uniform(0, 250, n)
    field_size = rng.uniform(0.2, 4.0, n)

    season = rng.choice(["A", "B"], size=n)
    cooperative_support = rng.integers(0, 2, size=n)

    season_effect = np.where(season == "A", 0.50, -0.20)

    noise = rng.normal(
        loc=0.0,
        scale=0.80,
        size=n
    )

    yield_tons = (
        0.004 * rainfall
        + 0.06 * soil_moisture
        + 0.02 * fertilizer
        + 1.30 * field_size
        - 0.25 * np.abs(temperature - 18)
        + 0.70 * cooperative_support
        + season_effect
        + noise
    )

    yield_tons = np.clip(yield_tons, 0.50, None)

    data = pd.DataFrame({
        "rainfall_mm": rainfall,
        "avg_temp_c": temperature,
        "soil_moisture_pct": soil_moisture,
        "fertilizer_kg": fertilizer,
        "field_size_ha": field_size,
        "season": season,
        "cooperative_support": cooperative_support,
        "yield_tons": yield_tons
    })

    # Classification target:
    # yield per hectare below the median is high risk.
    yield_per_hectare = data["yield_tons"] / data["field_size_ha"]
    median_yield_per_hectare = yield_per_hectare.median()

    data["high_risk"] = (
        yield_per_hectare < median_yield_per_hectare
    ).astype(int)

    # Deliberate missing values.
    m = round(0.05 * n)

    soil_missing_positions = rng.choice(
        n,
        size=m,
        replace=False
    )

    fertilizer_missing_positions = rng.choice(
        n,
        size=m,
        replace=False
    )

    data.loc[
        soil_missing_positions,
        "soil_moisture_pct"
    ] = np.nan

    data.loc[
        fertilizer_missing_positions,
        "fertilizer_kg"
    ] = np.nan

    # Append exactly three duplicate rows.
    duplicate_positions = rng.choice(
        len(data),
        size=3,
        replace=False
    )

    duplicates = data.iloc[duplicate_positions].copy()

    data = pd.concat(
        [data, duplicates],
        ignore_index=True
    )

    return data


def save_dataset(data: pd.DataFrame) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    data.to_csv(
        DATA_FILE,
        index=False,
        float_format="%.4f"
    )

    file_hash = hashlib.sha256(
        DATA_FILE.read_bytes()
    ).hexdigest()

    fingerprint_text = (
        f"Student ID: {STUDENT_ID}\n"
        f"Verification Code: {VERIFICATION_CODE}\n"
        f"Seed: {SEED}\n"
        f"Base Rows: {BASE_ROWS}\n"
        f"Final Rows: {len(data)}\n"
        f"SHA-256: {file_hash}\n"
    )

    FINGERPRINT_FILE.write_text(
        fingerprint_text,
        encoding="utf-8"
    )


def main() -> None:
    data = generate_dataset()

    save_dataset(data)

    print("=" * 60)
    print("MUSANZE HARVEST INTELLIGENCE SERVICE")
    print("=" * 60)
    print(f"Student ID:          {STUDENT_ID}")
    print(f"Verification Code:   {VERIFICATION_CODE}")
    print(f"Seed:                {SEED}")
    print(f"Base rows:           {BASE_ROWS}")
    print(f"Final rows:          {len(data)}")
    print(f"Dataset:             {DATA_FILE}")
    print(f"Fingerprint:         {FINGERPRINT_FILE}")
    print("=" * 60)


if __name__ == "__main__":
    main()
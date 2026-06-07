"""
Task 1.1 — Data Ingestion
Downloads AI4I 2020 Predictive Maintenance Dataset from UCI ML Repository,
validates schema and class distribution, saves to data/raw.parquet.
"""

import sys
from pathlib import Path

import pandas as pd
from ucimlrepo import fetch_ucirepo

DATA_DIR = Path(__file__).parent.parent / "data"
OUTPUT_PATH = DATA_DIR / "raw.parquet"

SENSOR_COLS = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]
TARGET_COL = "Machine failure"


def download_dataset() -> pd.DataFrame:
    """Fetch AI4I 2020 from UCI (id=601) and return a flat DataFrame."""
    print("Fetching AI4I 2020 Predictive Maintenance Dataset from UCI...")
    dataset = fetch_ucirepo(id=601)
    X = dataset.data.features
    y = dataset.data.targets

    df = pd.concat([X, y], axis=1)
    return df


def validate(df: pd.DataFrame) -> None:
    """Assert expected schema and class imbalance constraints."""
    missing = [c for c in SENSOR_COLS + [TARGET_COL] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")

    failure_rate = df[TARGET_COL].mean()
    print(f"Failure rate: {failure_rate:.4%}")
    assert failure_rate < 0.05, (
        f"Expected failure rate < 5%, got {failure_rate:.4%}. "
        "Check that the correct dataset was loaded."
    )

    print(f"Shape: {df.shape}")
    print(f"Dtypes:\n{df.dtypes}")
    print(f"\nClass distribution:\n{df[TARGET_COL].value_counts()}")


def main() -> pd.DataFrame:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    df = download_dataset()
    validate(df)

    df.to_parquet(OUTPUT_PATH, index=False)
    print(f"\nSaved → {OUTPUT_PATH}")
    return df


if __name__ == "__main__":
    main()

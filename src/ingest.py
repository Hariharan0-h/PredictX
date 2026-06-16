"""
Task 1.1 — Data Ingestion
Downloads AI4I 2020 Predictive Maintenance Dataset from UCI ML Repository,
validates schema and class distribution, saves to data/raw.parquet.
"""

import sys
from pathlib import Path

import pandas as pd
from ucimlrepo import fetch_ucirepo

sys.path.insert(0, str(Path(__file__).parent.parent))
from custom_logging import logger
from exception import CustomException

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
    """
    Downloads the AI4I 2020 Predictive Maintenance Dataset from  UCI ML Repository.
    Combines features and target variables into a single unified pandas Dataframe.
    
    """
    logger.info("Fetching AI4I 2020 Predictive Maintenance Dataset from UCI...")
    dataset = fetch_ucirepo(id=601)
    df = pd.concat([dataset.data.features, dataset.data.targets], axis=1)
    return df


def validate(df: pd.DataFrame) -> None:
    """
    Validates the schema,checks for missing expected columns,and verifies the baseline machine failure rate is 
    within acceptance limits (<5%).
    """
    missing = [c for c in SENSOR_COLS + [TARGET_COL] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")

    failure_rate = df[TARGET_COL].mean()
    logger.info(f"Failure rate: {failure_rate:.4%}")
    assert failure_rate < 0.05, (
        f"Expected failure rate < 5%, got {failure_rate:.4%}."
    )
    logger.info(f"Shape: {df.shape}")
    logger.info(f"Class distribution: {df[TARGET_COL].value_counts().to_dict()}")


def main() -> pd.DataFrame:
    """
    Execute the full data ingestion pipeline: creates data directories,
    downloads the raw telemetry dataset,validates it,and save it as parquet.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df = download_dataset()
    validate(df)
    df.to_parquet(OUTPUT_PATH, index=False)
    logger.info(f"Saved -> {OUTPUT_PATH}")
    return df


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        raise CustomException(e, sys)

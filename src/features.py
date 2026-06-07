"""
Task 1.2 — Rolling Window Feature Engineering
Computes rolling mean, std, variance (window=10) for each sensor column.
Input:  data/raw.parquet
Output: data/features_internal.parquet
"""

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parent.parent / "data"
INPUT_PATH = DATA_DIR / "raw.parquet"
OUTPUT_PATH = DATA_DIR / "features_internal.parquet"

SENSOR_COLS = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]
WINDOW = 10


def make_rolling_features(df: pd.DataFrame, window: int = WINDOW) -> pd.DataFrame:
    """
    Add rolling mean, std, variance for each sensor column.
    Sorts by existing row order (no timestamp column in AI4I dataset).
    Drops NaN rows introduced by the rolling window.
    """
    df = df.copy().reset_index(drop=True)

    for col in SENSOR_COLS:
        safe = col.replace(" ", "_").replace("[", "").replace("]", "").replace("/", "_")
        df[f"{safe}_roll_mean"] = df[col].rolling(window).mean()
        df[f"{safe}_roll_std"] = df[col].rolling(window).std()
        df[f"{safe}_roll_var"] = df[col].rolling(window).var()

    n_before = len(df)
    df = df.dropna().reset_index(drop=True)
    print(f"Dropped {n_before - len(df)} NaN rows (rolling warm-up). Remaining: {len(df)}")

    return df


def main() -> pd.DataFrame:
    df_raw = pd.read_parquet(INPUT_PATH)
    print(f"Loaded raw data: {df_raw.shape}")

    df_feat = make_rolling_features(df_raw)

    new_cols = [c for c in df_feat.columns if "_roll_" in c]
    print(f"Added {len(new_cols)} rolling feature columns.")

    df_feat.to_parquet(OUTPUT_PATH, index=False)
    print(f"Saved → {OUTPUT_PATH}")
    return df_feat


if __name__ == "__main__":
    main()

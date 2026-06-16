"""
Task 1.3 — Simulate & Merge External Context + Ablation Study
Input:  data/features_internal.parquet
Output: data/features_fused.parquet, results/ablation.json
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.pipeline import Pipeline
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent.parent))
from custom_logging import logger
from exception import CustomException

DATA_DIR = Path(__file__).parent.parent / "data"
RESULTS_DIR = Path(__file__).parent.parent / "results"
INPUT_PATH = DATA_DIR / "features_internal.parquet"
OUTPUT_PATH = DATA_DIR / "features_fused.parquet"

TARGET_COL = "Machine failure"
RANDOM_STATE = 42
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)


def simulate_external_context(n: int, seed: int = RANDOM_STATE) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "ambient_temp_ext": rng.normal(loc=20.0, scale=5.0, size=n),
        "factory_load":     rng.uniform(low=0.4, high=1.0, size=n),
    })


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c != TARGET_COL]


def run_cv(X: pd.DataFrame, y: pd.Series, label: str) -> float:
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LGBMClassifier(n_estimators=200, random_state=RANDOM_STATE, verbose=-1)),
    ])
    scores = cross_val_score(pipe, X, y, cv=CV, scoring="f1_macro")
    mean_f1 = scores.mean()
    logger.info(f"[{label}] Macro F1: mean={mean_f1:.4f} +/- {scores.std():.4f}")
    return float(mean_f1)


def main() -> pd.DataFrame:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    df_internal = pd.read_parquet(INPUT_PATH)
    logger.info(f"Loaded internal features: {df_internal.shape}")

    ctx = simulate_external_context(len(df_internal))
    df_fused = pd.concat([df_internal.reset_index(drop=True), ctx], axis=1)
    df_fused.to_parquet(OUTPUT_PATH, index=False)
    logger.info(f"Saved fused features -> {OUTPUT_PATH}")

    y = df_internal[TARGET_COL]
    logger.info("Running ablation study...")
    f1_internal = run_cv(df_internal[get_feature_cols(df_internal)], y, "internal-only")
    f1_fused    = run_cv(df_fused[get_feature_cols(df_fused)], y, "fused")
    delta = f1_fused - f1_internal
    logger.info(f"Macro F1 lift from external context: {delta:+.4f}")

    ablation = {"internal_only_macro_f1": f1_internal, "fused_macro_f1": f1_fused, "delta": delta}
    with open(RESULTS_DIR / "ablation.json", "w") as f:
        json.dump(ablation, f, indent=2)
    logger.info(f"Saved -> {RESULTS_DIR / 'ablation.json'}")
    return df_fused


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        raise CustomException(e, sys)

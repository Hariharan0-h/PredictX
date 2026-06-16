"""
Feature Drift Detection — PredictX
Compares the distribution of each sensor feature in incoming data against
the training baseline using the Kolmogorov-Smirnov test.

A drift warning is logged when any feature's KS p-value drops below the
significance threshold (default 0.05), indicating the incoming distribution
has shifted from what the model was trained on.

Usage:
    python src/drift.py --train data/features_fused.parquet --new path/to/new.csv
    python src/drift.py --train data/features_fused.parquet --new path/to/new.csv --alpha 0.01

Output: results/drift_report.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent))
from custom_logging import logger
from exception import CustomException

RESULTS_DIR = Path(__file__).parent.parent / "results"

SENSOR_COLS = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

DEFAULT_ALPHA = 0.05


def compute_drift(
    df_train: pd.DataFrame,
    df_new: pd.DataFrame,
    feature_cols: list[str],
    alpha: float = DEFAULT_ALPHA,
) -> dict:
    """
    Run KS test for each feature. Returns a report dict with per-feature
    statistics and an overall drift flag.
    """
    report: dict = {"alpha": alpha, "features": {}, "drifted_features": [], "drift_detected": False}

    for col in feature_cols:
        if col not in df_train.columns or col not in df_new.columns:
            logger.info(f"  Skipping '{col}' — not found in one or both datasets")
            continue

        ks_stat, p_value = stats.ks_2samp(
            df_train[col].dropna().values,
            df_new[col].dropna().values,
        )
        drifted = bool(p_value < alpha)
        report["features"][col] = {
            "ks_statistic": round(float(ks_stat), 6),
            "p_value":      round(float(p_value), 6),
            "drifted":      drifted,
        }
        if drifted:
            report["drifted_features"].append(col)
            logger.info(f"  ⚠  DRIFT: '{col}'  KS={ks_stat:.4f}  p={p_value:.4f}")
        else:
            logger.info(f"  ✓  OK:    '{col}'  KS={ks_stat:.4f}  p={p_value:.4f}")

    report["drift_detected"] = len(report["drifted_features"]) > 0
    return report


def run(train_path: str, new_path: str, alpha: float = DEFAULT_ALPHA) -> dict:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    df_train = pd.read_parquet(train_path) if train_path.endswith(".parquet") else pd.read_csv(train_path)
    df_new   = pd.read_csv(new_path) if new_path.endswith(".csv") else pd.read_parquet(new_path)

    logger.info(f"Drift check — train: {len(df_train)} rows, new: {len(df_new)} rows, alpha={alpha}")

    cols = [c for c in SENSOR_COLS if c in df_train.columns]
    report = compute_drift(df_train, df_new, cols, alpha)

    n_drifted = len(report["drifted_features"])
    if report["drift_detected"]:
        logger.info(f"Drift detected in {n_drifted}/{len(cols)} features: {report['drifted_features']}")
    else:
        logger.info(f"No drift detected across {len(cols)} features.")

    out_path = RESULTS_DIR / "drift_report.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Saved drift report -> {out_path}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="PredictX drift detection")
    parser.add_argument("--train", required=True, help="Path to training data (parquet or csv)")
    parser.add_argument("--new",   required=True, help="Path to incoming data (csv or parquet)")
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA,
                        help=f"KS test significance level (default {DEFAULT_ALPHA})")
    args = parser.parse_args()
    run(args.train, args.new, args.alpha)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        raise CustomException(e, sys)

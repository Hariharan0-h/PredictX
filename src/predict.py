"""
Inference Script — PredictX
Loads the trained pipeline + tuned decision threshold and scores new sensor data.

Usage:
    python src/predict.py --input path/to/sensors.csv
    python src/predict.py --input path/to/sensors.csv --threshold 0.35

Input CSV must contain the same feature columns used during training
(listed in results/cv_scores.json -> feature_cols).

Output: outputs/predictions.csv  with columns [prob_failure, predicted_failure]
"""

import argparse
import json
import sys
from pathlib import Path

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from custom_logging import logger
from exception import CustomException

MODELS_DIR  = Path(__file__).parent.parent / "models"
RESULTS_DIR = Path(__file__).parent.parent / "results"
OUTPUTS_DIR = Path(__file__).parent.parent / "outputs"

PIPELINE_PATH    = MODELS_DIR  / "pipeline.pkl"
CV_SCORES_PATH   = RESULTS_DIR / "cv_scores.json"
EVAL_METRICS_PATH = RESULTS_DIR / "eval_metrics.json"

DEFAULT_THRESHOLD = 0.5


def load_threshold(override: float | None) -> float:
    if override is not None:
        logger.info(f"Using CLI-supplied threshold: {override}")
        return override
    if EVAL_METRICS_PATH.exists():
        with open(EVAL_METRICS_PATH) as f:
            metrics = json.load(f)
        t = metrics.get("tuned_threshold", DEFAULT_THRESHOLD)
        logger.info(f"Loaded tuned threshold from eval_metrics.json: {t:.4f}")
        return t
    logger.info(f"eval_metrics.json not found — using default threshold {DEFAULT_THRESHOLD}")
    return DEFAULT_THRESHOLD


def load_feature_cols() -> list[str]:
    if not CV_SCORES_PATH.exists():
        raise FileNotFoundError(
            f"{CV_SCORES_PATH} not found. Run train.py first."
        )
    with open(CV_SCORES_PATH) as f:
        return json.load(f)["feature_cols"]


def predict(input_path: str, threshold_override: float | None = None) -> pd.DataFrame:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    if not PIPELINE_PATH.exists():
        raise FileNotFoundError(f"{PIPELINE_PATH} not found. Run train.py first.")

    df_input = pd.read_csv(input_path)
    logger.info(f"Loaded input: {df_input.shape[0]} rows from {input_path}")

    feature_cols = load_feature_cols()
    missing = [c for c in feature_cols if c not in df_input.columns]
    if missing:
        raise ValueError(f"Input CSV missing columns: {missing}")

    X = df_input[feature_cols]
    pipe = joblib.load(PIPELINE_PATH)
    logger.info("Pipeline loaded.")

    probs = pipe.predict_proba(X)[:, 1]
    threshold = load_threshold(threshold_override)
    preds = (probs >= threshold).astype(int)

    n_failures = int(preds.sum())
    logger.info(f"Threshold: {threshold:.4f} | Predicted failures: {n_failures}/{len(preds)}")

    out = df_input.copy()
    out["prob_failure"]      = probs.round(4)
    out["predicted_failure"] = preds

    out_path = OUTPUTS_DIR / "predictions.csv"
    out.to_csv(out_path, index=False)
    logger.info(f"Saved predictions -> {out_path}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="PredictX inference")
    parser.add_argument("--input",     required=True,  help="Path to input CSV")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Decision threshold (default: from eval_metrics.json)")
    args = parser.parse_args()
    predict(args.input, args.threshold)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        raise CustomException(e, sys)

"""
Task 1.5 — Noise Injection & Threshold Tuning
Input:  data/features_fused.parquet, models/pipeline.pkl, results/cv_scores.json
Output: outputs/pr_curve_noisy.png, results/eval_metrics.json
"""

import json
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import PrecisionRecallDisplay, classification_report, f1_score, precision_recall_curve
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent.parent))
from custom_logging import logger
from exception import CustomException

DATA_DIR    = Path(__file__).parent.parent / "data"
MODELS_DIR  = Path(__file__).parent.parent / "models"
RESULTS_DIR = Path(__file__).parent.parent / "results"
OUTPUTS_DIR = Path(__file__).parent.parent / "outputs"

INPUT_PATH    = DATA_DIR / "features_fused.parquet"
PIPELINE_PATH = MODELS_DIR / "pipeline.pkl"
CV_SCORES_PATH = RESULTS_DIR / "cv_scores.json"

TARGET_COL = "Machine failure"
SENSOR_COLS = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]
NOISE_SCALE  = 0.1
RANDOM_STATE = 42
TEST_SIZE    = 0.2


def inject_noise(X: pd.DataFrame, sensor_cols: list[str], scale: float, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    X_noisy = X.copy()
    for col in sensor_cols:
        if col in X_noisy.columns:
            sigma = X_noisy[col].std() * scale
            X_noisy[col] += rng.normal(0, sigma, size=len(X_noisy))
    return X_noisy


def find_best_threshold(y_true, y_scores) -> tuple[float, float]:
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    f1s = 2 * precision * recall / (precision + recall + 1e-9)
    best_idx = np.argmax(f1s[:-1])
    return float(thresholds[best_idx]), float(f1s[best_idx])


def main() -> dict:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(INPUT_PATH)
    with open(CV_SCORES_PATH) as f:
        feature_cols = json.load(f)["feature_cols"]

    X, y = df[feature_cols], df[TARGET_COL]
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    pipe = joblib.load(PIPELINE_PATH)
    X_test_noisy = inject_noise(X_test, SENSOR_COLS, NOISE_SCALE, seed=RANDOM_STATE)
    y_scores = pipe.predict_proba(X_test_noisy)[:, 1]

    f1_default = f1_score(y_test, (y_scores >= 0.5).astype(int), average="macro")
    best_threshold, _ = find_best_threshold(y_test, y_scores)
    f1_tuned = f1_score(y_test, (y_scores >= best_threshold).astype(int), average="macro")

    logger.info(f"Default threshold (0.5) — Macro F1: {f1_default:.4f}")
    logger.info(f"Tuned threshold ({best_threshold:.3f}) — Macro F1: {f1_tuned:.4f}")

    fig, ax = plt.subplots(figsize=(8, 6))
    PrecisionRecallDisplay.from_predictions(y_test, y_scores, ax=ax, name="LightGBM (noisy)")
    ax.axvline(x=best_threshold, color="red", linestyle="--",
               label=f"Best threshold={best_threshold:.3f}")
    ax.set_title("Precision-Recall Curve — Noisy Test Set")
    ax.legend()
    plot_path = OUTPUTS_DIR / "pr_curve_noisy.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved PR curve -> {plot_path}")

    y_pred_tuned = (y_scores >= best_threshold).astype(int)
    report = classification_report(y_test, y_pred_tuned, output_dict=True)
    metrics = {
        "default_threshold_macro_f1": float(f1_default),
        "tuned_threshold":            float(best_threshold),
        "tuned_threshold_macro_f1":   float(f1_tuned),
        "noise_scale":                NOISE_SCALE,
        "per_class": {
            "no_failure": report["0"],
            "failure":    report["1"],
        },
    }
    with open(RESULTS_DIR / "eval_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Saved eval metrics -> {RESULTS_DIR / 'eval_metrics.json'}")
    return metrics


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        raise CustomException(e, sys)

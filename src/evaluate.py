"""
Task 1.5 — Noise Injection & Threshold Tuning
Injects Gaussian noise (σ=0.1×feature_std) into sensor columns on the held-out
test set, plots the Precision-Recall curve, and finds the threshold that maximises F1.
Input:  data/features_fused.parquet, models/pipeline.pkl, results/cv_scores.json
Output: outputs/pr_curve_noisy.png
        results/eval_metrics.json
"""

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    PrecisionRecallDisplay,
    f1_score,
    precision_recall_curve,
)
from sklearn.model_selection import train_test_split

DATA_DIR = Path(__file__).parent.parent / "data"
MODELS_DIR = Path(__file__).parent.parent / "models"
RESULTS_DIR = Path(__file__).parent.parent / "results"
OUTPUTS_DIR = Path(__file__).parent.parent / "outputs"

INPUT_PATH = DATA_DIR / "features_fused.parquet"
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
NOISE_SCALE = 0.1
RANDOM_STATE = 42
TEST_SIZE = 0.2


def inject_noise(X: pd.DataFrame, sensor_cols: list[str], scale: float, seed: int) -> pd.DataFrame:
    """Add Gaussian noise σ = scale × per-column std to sensor columns only."""
    rng = np.random.default_rng(seed)
    X_noisy = X.copy()
    for col in sensor_cols:
        if col in X_noisy.columns:
            sigma = X_noisy[col].std() * scale
            X_noisy[col] += rng.normal(0, sigma, size=len(X_noisy))
    return X_noisy


def find_best_threshold(y_true, y_scores) -> tuple[float, float]:
    """Return (threshold, f1) that maximises F1 on the PR curve."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    f1s = 2 * precision * recall / (precision + recall + 1e-9)
    best_idx = np.argmax(f1s[:-1])  # last entry has no matching threshold
    return float(thresholds[best_idx]), float(f1s[best_idx])


def main() -> dict:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(INPUT_PATH)
    with open(CV_SCORES_PATH) as f:
        cv_info = json.load(f)
    feature_cols = cv_info["feature_cols"]

    X = df[feature_cols]
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    pipe = joblib.load(PIPELINE_PATH)

    # Inject noise into test set sensor columns only
    X_test_noisy = inject_noise(X_test, SENSOR_COLS, NOISE_SCALE, seed=RANDOM_STATE)

    y_scores = pipe.predict_proba(X_test_noisy)[:, 1]

    # Default threshold (0.5)
    y_pred_default = (y_scores >= 0.5).astype(int)
    f1_default = f1_score(y_test, y_pred_default, average="macro")

    # Tuned threshold
    best_threshold, f1_tuned = find_best_threshold(y_test, y_scores)
    y_pred_tuned = (y_scores >= best_threshold).astype(int)
    f1_tuned_macro = f1_score(y_test, y_pred_tuned, average="macro")

    print(f"Default threshold (0.5) — Macro F1: {f1_default:.4f}")
    print(f"Tuned threshold  ({best_threshold:.3f}) — Macro F1: {f1_tuned_macro:.4f}")

    # PR curve plot
    fig, ax = plt.subplots(figsize=(8, 6))
    PrecisionRecallDisplay.from_predictions(y_test, y_scores, ax=ax, name="LightGBM (noisy)")
    ax.axvline(x=best_threshold, color="red", linestyle="--",
               label=f"Best threshold={best_threshold:.3f}")
    ax.set_title("Precision-Recall Curve — Noisy Test Set")
    ax.legend()
    plot_path = OUTPUTS_DIR / "pr_curve_noisy.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved PR curve → {plot_path}")

    metrics = {
        "default_threshold_macro_f1": float(f1_default),
        "tuned_threshold": float(best_threshold),
        "tuned_threshold_macro_f1": float(f1_tuned_macro),
        "noise_scale": NOISE_SCALE,
    }
    with open(RESULTS_DIR / "eval_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved eval metrics → {RESULTS_DIR / 'eval_metrics.json'}")

    return metrics


if __name__ == "__main__":
    main()

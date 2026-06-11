"""
SHAP Explainability
Loads trained LightGBM pipeline, computes SHAP values via TreeExplainer,
and saves summary bar + beeswarm plots plus top-10 feature JSON.

Prerequisites: run train.py first so models/pipeline.pkl exists.
Output: outputs/shap_summary_bar.png
        outputs/shap_beeswarm.png
        results/shap_top10.json
"""

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.model_selection import train_test_split

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from custom_logging import logger

DATA_DIR = Path(__file__).parent.parent / "data"
MODELS_DIR = Path(__file__).parent.parent / "models"
RESULTS_DIR = Path(__file__).parent.parent / "results"
OUTPUTS_DIR = Path(__file__).parent.parent / "outputs"

INPUT_PATH = DATA_DIR / "features_fused.parquet"
PIPELINE_PATH = MODELS_DIR / "pipeline.pkl"
CV_SCORES_PATH = RESULTS_DIR / "cv_scores.json"

TARGET_COL = "Machine failure"
RANDOM_STATE = 42
TEST_SIZE = 0.2
SHAP_SAMPLE = None


def extract_lgbm_and_transform(pipe, X: pd.DataFrame):
    """
    Run all pipeline steps except the classifier, returning the
    transformed feature matrix LightGBM sees at inference.
    SMOTE has no transform at inference — skipped silently.
    """
    X_t = X.copy()
    for name, step in pipe.steps[:-1]:
        if hasattr(step, "transform"):
            X_t = step.transform(X_t)
    return X_t


def main() -> dict:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(INPUT_PATH)
    with open(CV_SCORES_PATH) as f:
        cv_info = json.load(f)
    feature_cols = cv_info["feature_cols"]

    X = df[feature_cols]
    y = df[TARGET_COL]

    _, X_test, _, _ = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    pipe = joblib.load(PIPELINE_PATH)
    clf = pipe.named_steps["clf"]

    X_test_transformed = extract_lgbm_and_transform(pipe, X_test)

    if not isinstance(X_test_transformed, pd.DataFrame):
        X_test_transformed = pd.DataFrame(X_test_transformed, columns=feature_cols)

    if SHAP_SAMPLE and len(X_test_transformed) > SHAP_SAMPLE:
        X_test_transformed = X_test_transformed.sample(SHAP_SAMPLE, random_state=RANDOM_STATE)

    logger.info(f"Computing SHAP values for {len(X_test_transformed)} samples...")
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X_test_transformed)

    if isinstance(shap_values, list):
        sv = shap_values[1]
    else:
        sv = shap_values

    mean_abs_shap = np.abs(sv).mean(axis=0)
    top10_idx = np.argsort(mean_abs_shap)[::-1][:10]
    top10 = {feature_cols[i]: float(mean_abs_shap[i]) for i in top10_idx}

    logger.info("Top-10 features by mean |SHAP|:")
    for feat, val in top10.items():
        logger.info(f"  {val:.4f}  {feat}")

    fig, ax = plt.subplots(figsize=(10, 6))
    shap.summary_plot(sv, X_test_transformed, plot_type="bar", show=False)
    plt.title("SHAP Feature Importance (mean |SHAP value|)")
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / "shap_summary_bar.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved -> {OUTPUTS_DIR / 'shap_summary_bar.png'}")

    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(sv, X_test_transformed, show=False)
    plt.title("SHAP Beeswarm — Feature Impact on Failure Prediction")
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / "shap_beeswarm.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved -> {OUTPUTS_DIR / 'shap_beeswarm.png'}")

    with open(RESULTS_DIR / "shap_top10.json", "w") as f:
        json.dump(top10, f, indent=2)
    logger.info(f"Saved -> {RESULTS_DIR / 'shap_top10.json'}")

    return top10


if __name__ == "__main__":
    main()

"""
SHAP Explainability
Loads the trained LightGBM pipeline, computes SHAP values via TreeExplainer
on the held-out test set (clean, no noise), and saves:
  - outputs/shap_summary_bar.png   — mean |SHAP| bar chart (global importance)
  - outputs/shap_beeswarm.png      — beeswarm plot (value distribution per feature)
  - results/shap_top10.json        — top-10 features by mean |SHAP|

Prerequisites: run train.py first so models/pipeline.pkl exists.
"""

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.model_selection import train_test_split

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
# Subsample for SHAP speed (TreeExplainer on 200+ samples is fast; keep full for accuracy)
SHAP_SAMPLE = None  # set to e.g. 500 to speed up on large datasets


def extract_lgbm_and_transform(pipe, X: pd.DataFrame):
    """
    Run all pipeline steps except the final classifier, returning the
    transformed feature matrix that LightGBM actually sees.
    SMOTE is a fit-only step — it has no transform for inference, so we
    iterate through steps and skip resampling-only steps.
    """
    X_t = X.copy()
    for name, step in pipe.steps[:-1]:  # all but classifier
        if hasattr(step, "transform"):
            X_t = step.transform(X_t)
        # SMOTE has no transform method at inference time — skip silently
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
    clf = pipe.named_steps["clf"]  # LGBMClassifier

    # Transform test set through pre-processing steps (scaler only at inference)
    X_test_transformed = extract_lgbm_and_transform(pipe, X_test)

    # Wrap as DataFrame so SHAP preserves feature names
    if not isinstance(X_test_transformed, pd.DataFrame):
        X_test_transformed = pd.DataFrame(X_test_transformed, columns=feature_cols)

    if SHAP_SAMPLE and len(X_test_transformed) > SHAP_SAMPLE:
        X_test_transformed = X_test_transformed.sample(SHAP_SAMPLE, random_state=RANDOM_STATE)

    print(f"Computing SHAP values for {len(X_test_transformed)} samples...")
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X_test_transformed)

    # For binary classification LightGBM, shap_values is a list [class0, class1]
    # or a 2D array depending on SHAP version — normalise to class-1 values
    if isinstance(shap_values, list):
        sv = shap_values[1]
    else:
        sv = shap_values

    mean_abs_shap = np.abs(sv).mean(axis=0)
    top10_idx = np.argsort(mean_abs_shap)[::-1][:10]
    top10 = {feature_cols[i]: float(mean_abs_shap[i]) for i in top10_idx}

    print("\nTop-10 features by mean |SHAP|:")
    for feat, val in top10.items():
        print(f"  {val:.4f}  {feat}")

    # --- Summary bar chart ---
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.summary_plot(sv, X_test_transformed, plot_type="bar", show=False)
    plt.title("SHAP Feature Importance (mean |SHAP value|)")
    plt.tight_layout()
    bar_path = OUTPUTS_DIR / "shap_summary_bar.png"
    plt.savefig(bar_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {bar_path}")

    # --- Beeswarm plot ---
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(sv, X_test_transformed, show=False)
    plt.title("SHAP Beeswarm — Feature Impact on Failure Prediction")
    plt.tight_layout()
    bee_path = OUTPUTS_DIR / "shap_beeswarm.png"
    plt.savefig(bee_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {bee_path}")

    with open(RESULTS_DIR / "shap_top10.json", "w") as f:
        json.dump(top10, f, indent=2)
    print(f"Saved → {RESULTS_DIR / 'shap_top10.json'}")

    return top10


if __name__ == "__main__":
    main()

"""
Task 1.4 — SMOTE-inside-CV LightGBM Pipeline
Input:  data/features_fused.parquet
Output: models/pipeline.pkl, results/cv_scores.json
"""

import json
import sys
from pathlib import Path

import joblib
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent.parent))
from custom_logging import logger
from exception import CustomException

DATA_DIR    = Path(__file__).parent.parent / "data"
MODELS_DIR  = Path(__file__).parent.parent / "models"
RESULTS_DIR = Path(__file__).parent.parent / "results"
INPUT_PATH  = DATA_DIR / "features_fused.parquet"

TARGET_COL       = "Machine failure"
RANDOM_STATE     = 42
F1_THRESHOLD     = 0.85
BEST_PARAMS_PATH = RESULTS_DIR / "best_params.json"

DEFAULT_CLF_PARAMS = {"n_estimators": 300, "learning_rate": 0.05, "num_leaves": 31}


def load_clf_params() -> dict:
    if BEST_PARAMS_PATH.exists():
        with open(BEST_PARAMS_PATH) as f:
            data = json.load(f)
        logger.info(f"Loaded tuned params (best CV F1={data['macro_f1']:.4f})")
        return data["params"]
    logger.info("No best_params.json found — using default hyperparameters.")
    return DEFAULT_CLF_PARAMS


def build_pipeline() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("smote",  SMOTE(random_state=RANDOM_STATE)),
        ("clf",    LGBMClassifier(**load_clf_params(), random_state=RANDOM_STATE, verbose=-1)),
    ])


def main() -> dict:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(INPUT_PATH)
    feature_cols = [c for c in df.columns if c != TARGET_COL]
    X, y = df[feature_cols], df[TARGET_COL]

    logger.info(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    logger.info(f"Class distribution: {y.value_counts().to_dict()}")

    pipe = build_pipeline()
    cv   = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    logger.info("Running 5-fold stratified CV with SMOTE inside folds...")
    scores = cross_val_score(pipe, X, y, cv=cv, scoring="f1_macro", n_jobs=-1)

    for i, s in enumerate(scores, 1):
        logger.info(f"  Fold {i}: {s:.4f}")
    mean_f1, std_f1 = scores.mean(), scores.std()
    logger.info(f"Macro F1: {mean_f1:.4f} +/- {std_f1:.4f}")

    assert mean_f1 >= F1_THRESHOLD, (
        f"Macro F1 {mean_f1:.4f} below threshold {F1_THRESHOLD}. "
        f"Run tune.py or review feature engineering."
    )
    logger.info(f"Macro F1 >= {F1_THRESHOLD} — threshold met.")

    pipe.fit(X, y)
    joblib.dump(pipe, MODELS_DIR / "pipeline.pkl")
    logger.info(f"Saved pipeline -> {MODELS_DIR / 'pipeline.pkl'}")

    results = {
        "per_fold_f1":   scores.tolist(),
        "mean_macro_f1": float(mean_f1),
        "std_macro_f1":  float(std_f1),
        "feature_cols":  feature_cols,
    }
    with open(RESULTS_DIR / "cv_scores.json", "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved CV scores -> {RESULTS_DIR / 'cv_scores.json'}")
    return results


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        raise CustomException(e, sys)

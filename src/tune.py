"""
Hyperparameter Tuning — Optuna sweep over LightGBM inside the SMOTE pipeline.
Optimises Macro F1 via 5-fold stratified CV (same setup as train.py).
Best params are injected back into train.py's build_pipeline() via results/best_params.json.

Usage:
    python src/tune.py              # runs 50 trials (default)
    python src/tune.py --trials 100

Input:  data/features_fused.parquet
Output: results/best_params.json
        results/tune_study.pkl      (Optuna study — gitignored)
"""

import argparse
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError:
    raise ImportError("Install optuna first: pip install optuna")

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from custom_logging import logger
from exception import CustomException

DATA_DIR = Path(__file__).parent.parent / "data"
RESULTS_DIR = Path(__file__).parent.parent / "results"
INPUT_PATH = DATA_DIR / "features_fused.parquet"

TARGET_COL = "Machine failure"
RANDOM_STATE = 42
N_CV_SPLITS = 5
DEFAULT_TRIALS = 50
TIMEOUT_SECONDS= 3600


def load_data() -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_parquet(INPUT_PATH)
    feature_cols = [c for c in df.columns if c != TARGET_COL]
    return df[feature_cols], df[TARGET_COL]


def objective(trial, X: pd.DataFrame, y: pd.Series) -> float:
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 600),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 16, 128),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 1.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 1.0, log=True),
    }

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("smote", SMOTE(random_state=RANDOM_STATE)),
        ("clf", LGBMClassifier(
            **params,
            random_state=RANDOM_STATE,
            verbose=-1,
            n_jobs=1,
        )),
    ])

    cv = StratifiedKFold(n_splits=N_CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(pipe, X, y, cv=cv, scoring="f1_macro")
    return float(scores.mean())


def main(n_trials: int = DEFAULT_TRIALS) -> dict:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    X, y = load_data()
    logger.info(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    logger.info(f"Running Optuna sweep — {n_trials} trials...\n")

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
    )
    study.optimize(
        lambda trial: objective(trial, X, y),
        n_trials=n_trials,
        n_jobs=-1,
        show_progress_bar=True,
    )

    best = study.best_trial
    logger.info(f"\nBest Macro F1: {best.value:.4f}")
    logger.info(f"Best params:   {best.params}")

    best_params = {
        "macro_f1": best.value,
        "params": best.params,
        "n_trials": n_trials,
    }
    with open(RESULTS_DIR / "best_params.json", "w") as f:
        json.dump(best_params, f, indent=2)
    logger.info(f"Saved → {RESULTS_DIR / 'best_params.json'}")

    # Persist study for post-hoc analysis (gitignored via models/ pattern)
    joblib.dump(study, RESULTS_DIR / "tune_study.pkl")

    return best_params


if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS)
        args = parser.parse_args()
        main(n_trials=args.trials)
    except Exception as e:
        raise CustomException(e, sys)

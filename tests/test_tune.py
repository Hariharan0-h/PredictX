"""Tests for hyperparameter tuning (src/tune.py)."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd
import pytest
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)

from tune import objective


@pytest.fixture
def small_dataset():
    """Minimal binary dataset with class imbalance (~5% positives)."""
    rng = np.random.default_rng(0)
    n = 300
    X = pd.DataFrame(rng.normal(size=(n, 6)), columns=[f"f{i}" for i in range(6)])
    y = pd.Series((rng.random(n) > 0.95).astype(int))
    return X, y


def test_objective_returns_float(small_dataset):
    X, y = small_dataset
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, X, y), n_trials=1)
    assert isinstance(study.best_value, float)


def test_objective_f1_in_valid_range(small_dataset):
    X, y = small_dataset
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, X, y), n_trials=2)
    assert 0.0 <= study.best_value <= 1.0


def test_study_records_all_params(small_dataset):
    X, y = small_dataset
    expected_params = {
        "n_estimators", "learning_rate", "num_leaves", "max_depth",
        "min_child_samples", "subsample", "colsample_bytree",
        "reg_alpha", "reg_lambda",
    }
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, X, y), n_trials=1)
    assert expected_params == set(study.best_trial.params.keys())


def test_multiple_trials_maximise(small_dataset):
    """Best value after 3 trials should be >= value of first trial."""
    X, y = small_dataset
    trial_values = []
    study = optuna.create_study(direction="maximize")

    def cb(study, trial):
        trial_values.append(trial.value)

    study.optimize(lambda trial: objective(trial, X, y), n_trials=3, callbacks=[cb])
    assert study.best_value == max(trial_values)

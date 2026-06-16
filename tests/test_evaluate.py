"""Tests for noise injection and threshold tuning (src/evaluate.py)."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd
import pytest

from evaluate import find_best_threshold, inject_noise

SENSOR_COLS = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]


@pytest.fixture
def sample_X():
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {col: rng.normal(loc=300, scale=10, size=200) for col in SENSOR_COLS}
    )


# --- inject_noise ---

def test_noise_only_affects_sensor_cols(sample_X):
    extra_col = "factory_load"
    df = sample_X.copy()
    df[extra_col] = 0.7  # non-sensor column, constant

    noisy = inject_noise(df, SENSOR_COLS, scale=0.1, seed=0)

    # Non-sensor column must be unchanged
    pd.testing.assert_series_equal(df[extra_col], noisy[extra_col])

    # At least one sensor column must have changed
    changed = any(not df[c].equals(noisy[c]) for c in SENSOR_COLS)
    assert changed, "Noise injection did not modify any sensor column"


def test_noise_preserves_shape(sample_X):
    noisy = inject_noise(sample_X, SENSOR_COLS, scale=0.1, seed=0)
    assert noisy.shape == sample_X.shape


def test_noise_magnitude_is_small(sample_X):
    """Mean absolute noise should be << raw signal magnitude."""
    noisy = inject_noise(sample_X, SENSOR_COLS, scale=0.1, seed=0)
    for col in SENSOR_COLS:
        delta = (noisy[col] - sample_X[col]).abs().mean()
        raw_std = sample_X[col].std()
        assert delta < raw_std, (
            f"Noise on '{col}' larger than expected: delta={delta:.3f}, raw_std={raw_std:.3f}"
        )


def test_noise_is_reproducible(sample_X):
    n1 = inject_noise(sample_X, SENSOR_COLS, scale=0.1, seed=42)
    n2 = inject_noise(sample_X, SENSOR_COLS, scale=0.1, seed=42)
    pd.testing.assert_frame_equal(n1, n2)


def test_zero_scale_produces_no_change(sample_X):
    noisy = inject_noise(sample_X, SENSOR_COLS, scale=0.0, seed=0)
    pd.testing.assert_frame_equal(sample_X, noisy)


# --- find_best_threshold ---

def test_threshold_in_unit_interval():
    rng = np.random.default_rng(1)
    y_true = rng.integers(0, 2, size=300)
    y_scores = rng.random(300)
    threshold, _ = find_best_threshold(y_true, y_scores)
    assert 0.0 <= threshold <= 1.0


def test_tuned_f1_ge_default_f1():
    """Tuned threshold F1 should be >= default (0.5) F1 on the same data."""
    from sklearn.metrics import f1_score

    rng = np.random.default_rng(7)
    y_true = rng.integers(0, 2, size=500)
    # Skew scores toward true class to give the tuner something to work with
    y_scores = np.where(y_true == 1, rng.uniform(0.4, 1.0, 500), rng.uniform(0.0, 0.6, 500))

    threshold, _ = find_best_threshold(y_true, y_scores)
    f1_default = f1_score(y_true, (y_scores >= 0.5).astype(int), average="macro")
    f1_tuned = f1_score(y_true, (y_scores >= threshold).astype(int), average="macro")

    assert f1_tuned >= f1_default - 1e-6  # allow floating point tolerance

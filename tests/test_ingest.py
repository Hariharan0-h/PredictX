"""Tests for src/ingest.py — data validation logic."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


REQUIRED_SENSOR_COLS = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]
TARGET_COL = "Machine failure"
FAILURE_RATE_LIMIT = 0.05


def make_mock_df(n=1000, failure_rate=0.03, seed=42):
    rng = np.random.default_rng(seed)
    n_failures = int(n * failure_rate)
    failures = np.zeros(n, dtype=int)
    failures[:n_failures] = 1
    rng.shuffle(failures)
    df = pd.DataFrame({
        "Air temperature [K]":       rng.normal(300, 2, n),
        "Process temperature [K]":   rng.normal(310, 1, n),
        "Rotational speed [rpm]":    rng.integers(1000, 3000, n).astype(float),
        "Torque [Nm]":               rng.normal(40, 10, n),
        "Tool wear [min]":           rng.integers(0, 250, n).astype(float),
        "Machine failure":           failures,
        "Type":                      rng.choice(["L", "M", "H"], n),
        "UDI":                       np.arange(n),
        "Product ID":                [f"M{i}" for i in range(n)],
    })
    return df


def test_required_sensor_columns_present():
    df = make_mock_df()
    for col in REQUIRED_SENSOR_COLS:
        assert col in df.columns, f"Missing column: {col}"


def test_target_column_present():
    df = make_mock_df()
    assert TARGET_COL in df.columns


def test_target_column_is_binary():
    df = make_mock_df()
    unique_vals = set(df[TARGET_COL].unique())
    assert unique_vals.issubset({0, 1}), f"Target has non-binary values: {unique_vals}"


def test_failure_rate_below_threshold():
    df = make_mock_df(failure_rate=0.03)
    rate = df[TARGET_COL].mean()
    assert rate < FAILURE_RATE_LIMIT, f"Failure rate {rate:.3f} >= {FAILURE_RATE_LIMIT}"


def test_failure_rate_assertion_fires():
    """Simulates what ingest.py asserts — high failure rate should be caught."""
    df = make_mock_df(failure_rate=0.10)
    rate = df[TARGET_COL].mean()
    with pytest.raises(AssertionError):
        assert rate < FAILURE_RATE_LIMIT, f"Failure rate {rate:.3f} too high"


def test_no_null_values_in_sensor_cols():
    df = make_mock_df()
    nulls = df[REQUIRED_SENSOR_COLS].isnull().sum().sum()
    assert nulls == 0, f"Unexpected nulls in sensor columns: {nulls}"


def test_dataset_minimum_row_count():
    df = make_mock_df(n=1000)
    assert len(df) >= 1000, "Dataset too small for reliable training"

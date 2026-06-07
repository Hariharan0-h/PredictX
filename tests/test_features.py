"""Tests for rolling window feature engineering (src/features.py)."""

import numpy as np
import pandas as pd
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from features import SENSOR_COLS, WINDOW, make_rolling_features


@pytest.fixture
def sample_df():
    n = 50
    rng = np.random.default_rng(0)
    data = {col: rng.normal(size=n) for col in SENSOR_COLS}
    data["Machine failure"] = rng.integers(0, 2, size=n)
    data["Type"] = "M"
    return pd.DataFrame(data)


def test_output_has_rolling_cols(sample_df):
    out = make_rolling_features(sample_df, window=WINDOW)
    expected_suffixes = ["_roll_mean", "_roll_std", "_roll_var"]
    for col in SENSOR_COLS:
        safe = col.replace(" ", "_").replace("[", "").replace("]", "").replace("/", "_")
        for suf in expected_suffixes:
            assert f"{safe}{suf}" in out.columns, f"Missing column: {safe}{suf}"


def test_no_nans_in_output(sample_df):
    out = make_rolling_features(sample_df, window=WINDOW)
    assert out.isnull().sum().sum() == 0, "Output contains NaN values after rolling"


def test_row_count_reduced_by_window(sample_df):
    out = make_rolling_features(sample_df, window=WINDOW)
    assert len(out) == len(sample_df) - WINDOW + 1


def test_original_cols_preserved(sample_df):
    out = make_rolling_features(sample_df, window=WINDOW)
    for col in SENSOR_COLS + ["Machine failure"]:
        assert col in out.columns

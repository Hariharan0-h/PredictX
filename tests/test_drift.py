"""Tests for src/drift.py — KS-test feature drift detection."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from drift import compute_drift, DEFAULT_ALPHA


def make_df(means: dict, n: int = 500, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({col: rng.normal(mu, 1, n) for col, mu in means.items()})


COLS = ["Air temperature [K]", "Process temperature [K]", "Rotational speed [rpm]"]


def test_no_drift_on_identical_distribution():
    df = make_df({c: 0.0 for c in COLS}, seed=1)
    df2 = make_df({c: 0.0 for c in COLS}, seed=2)
    report = compute_drift(df, df2, COLS)
    assert not report["drift_detected"], "Identical distributions should not flag drift"


def test_drift_detected_on_large_shift():
    df_train = make_df({c: 0.0 for c in COLS})
    df_new   = make_df({c: 10.0 for c in COLS})   # massive shift
    report = compute_drift(df_train, df_new, COLS)
    assert report["drift_detected"], "Large shift must be detected as drift"
    assert len(report["drifted_features"]) == len(COLS)


def test_drift_report_keys_present():
    df = make_df({c: 0.0 for c in COLS})
    report = compute_drift(df, df.copy(), COLS)
    assert "alpha" in report
    assert "features" in report
    assert "drifted_features" in report
    assert "drift_detected" in report


def test_per_feature_report_keys():
    df = make_df({c: 0.0 for c in COLS})
    report = compute_drift(df, df.copy(), COLS)
    for col in COLS:
        feat = report["features"][col]
        assert "ks_statistic" in feat
        assert "p_value" in feat
        assert "drifted" in feat


def test_ks_statistic_range():
    df = make_df({c: 0.0 for c in COLS})
    report = compute_drift(df, df.copy(), COLS)
    for col, feat in report["features"].items():
        assert 0.0 <= feat["ks_statistic"] <= 1.0


def test_partial_drift_only_flags_shifted_col():
    df_train = make_df({c: 0.0 for c in COLS})
    df_new   = make_df({c: 0.0 for c in COLS})
    # Shift only the first column heavily
    df_new[COLS[0]] += 15.0
    report = compute_drift(df_train, df_new, COLS, alpha=DEFAULT_ALPHA)
    assert COLS[0] in report["drifted_features"]
    # Other columns should not drift
    for col in COLS[1:]:
        assert col not in report["drifted_features"]


def test_alpha_controls_sensitivity():
    df_train = make_df({c: 0.0 for c in COLS})
    df_new   = make_df({c: 0.5 for c in COLS})   # mild shift
    strict  = compute_drift(df_train, df_new, COLS, alpha=0.5)
    lenient = compute_drift(df_train, df_new, COLS, alpha=0.001)
    # Stricter alpha → more drift flags; lenient alpha → fewer
    assert len(strict["drifted_features"]) >= len(lenient["drifted_features"])

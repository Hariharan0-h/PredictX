"""Tests for src/predict.py — inference script."""

import json
import sys
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from lightgbm import LGBMClassifier
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from predict import predict, load_threshold, DEFAULT_THRESHOLD


@pytest.fixture
def tiny_pipeline_and_data(tmp_path):
    rng = np.random.default_rng(0)
    n = 300
    cols = [f"f{i}" for i in range(5)]
    X = pd.DataFrame(rng.normal(size=(n, 5)), columns=cols)
    y = pd.Series((rng.random(n) > 0.85).astype(int))

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("smote", SMOTE(random_state=0)),
        ("clf", LGBMClassifier(n_estimators=10, random_state=0, verbose=-1)),
    ])
    pipe.fit(X, y)

    model_dir = tmp_path / "models"
    model_dir.mkdir()
    joblib.dump(pipe, model_dir / "pipeline.pkl")

    results_dir = tmp_path / "results"
    results_dir.mkdir()
    cv_scores = {"feature_cols": cols}
    (results_dir / "cv_scores.json").write_text(json.dumps(cv_scores))

    input_csv = tmp_path / "sensors.csv"
    X.to_csv(input_csv, index=False)

    return tmp_path, pipe, cols, input_csv


def test_predict_output_has_required_columns(tiny_pipeline_and_data, monkeypatch):
    tmp_path, _, cols, input_csv = tiny_pipeline_and_data
    _patch_paths(monkeypatch, tmp_path)
    out = predict(str(input_csv))
    assert "prob_failure" in out.columns
    assert "predicted_failure" in out.columns


def test_predict_output_row_count_matches_input(tiny_pipeline_and_data, monkeypatch):
    tmp_path, _, cols, input_csv = tiny_pipeline_and_data
    _patch_paths(monkeypatch, tmp_path)
    df_in = pd.read_csv(input_csv)
    out = predict(str(input_csv))
    assert len(out) == len(df_in)


def test_predict_probabilities_in_unit_interval(tiny_pipeline_and_data, monkeypatch):
    tmp_path, _, _, input_csv = tiny_pipeline_and_data
    _patch_paths(monkeypatch, tmp_path)
    out = predict(str(input_csv))
    assert out["prob_failure"].between(0, 1).all()


def test_predict_binary_predictions(tiny_pipeline_and_data, monkeypatch):
    tmp_path, _, _, input_csv = tiny_pipeline_and_data
    _patch_paths(monkeypatch, tmp_path)
    out = predict(str(input_csv))
    assert set(out["predicted_failure"].unique()).issubset({0, 1})


def test_predict_missing_columns_raises(tiny_pipeline_and_data, monkeypatch, tmp_path):
    base, _, cols, _ = tiny_pipeline_and_data
    _patch_paths(monkeypatch, base)
    bad_csv = tmp_path / "bad.csv"
    pd.DataFrame({"wrong_col": [1, 2, 3]}).to_csv(bad_csv, index=False)
    with pytest.raises(ValueError, match="missing columns"):
        predict(str(bad_csv))


def test_load_threshold_uses_default_when_no_file(tmp_path, monkeypatch):
    import predict as predict_module
    monkeypatch.setattr(predict_module, "EVAL_METRICS_PATH", tmp_path / "nonexistent.json")
    t = load_threshold(None)
    assert t == DEFAULT_THRESHOLD


def test_load_threshold_reads_from_metrics(tmp_path, monkeypatch):
    import predict as predict_module
    p = tmp_path / "eval_metrics.json"
    p.write_text(json.dumps({"tuned_threshold": 0.37}))
    monkeypatch.setattr(predict_module, "EVAL_METRICS_PATH", p)
    assert load_threshold(None) == pytest.approx(0.37)


def test_load_threshold_cli_override_takes_precedence(tmp_path, monkeypatch):
    import predict as predict_module
    p = tmp_path / "eval_metrics.json"
    p.write_text(json.dumps({"tuned_threshold": 0.37}))
    monkeypatch.setattr(predict_module, "EVAL_METRICS_PATH", p)
    assert load_threshold(0.6) == pytest.approx(0.6)


# --- helpers ---

def _patch_paths(monkeypatch, tmp_path):
    import predict as predict_module
    monkeypatch.setattr(predict_module, "PIPELINE_PATH",   tmp_path / "models" / "pipeline.pkl")
    monkeypatch.setattr(predict_module, "CV_SCORES_PATH",  tmp_path / "results" / "cv_scores.json")
    monkeypatch.setattr(predict_module, "EVAL_METRICS_PATH", tmp_path / "results" / "eval_metrics.json")
    monkeypatch.setattr(predict_module, "OUTPUTS_DIR",     tmp_path / "outputs")

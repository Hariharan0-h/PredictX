"""Tests for src/train.py — pipeline structure and param loading."""

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from imblearn.pipeline import Pipeline as ImbPipeline
from lightgbm import LGBMClassifier
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from train import build_pipeline, load_clf_params, DEFAULT_CLF_PARAMS


def test_build_pipeline_returns_imblearn_pipeline():
    """Must use imblearn.Pipeline — not sklearn — to prevent SMOTE leakage."""
    pipe = build_pipeline()
    assert isinstance(pipe, ImbPipeline), (
        "Pipeline must be imblearn.Pipeline to keep SMOTE inside CV folds"
    )


def test_pipeline_has_three_steps():
    pipe = build_pipeline()
    assert len(pipe.steps) == 3, f"Expected 3 steps, got {len(pipe.steps)}"


def test_pipeline_step_names():
    pipe = build_pipeline()
    names = [name for name, _ in pipe.steps]
    assert names == ["scaler", "smote", "clf"]


def test_pipeline_scaler_is_standard_scaler():
    pipe = build_pipeline()
    assert isinstance(pipe.named_steps["scaler"], StandardScaler)


def test_pipeline_clf_is_lgbm():
    pipe = build_pipeline()
    assert isinstance(pipe.named_steps["clf"], LGBMClassifier)


def test_load_clf_params_returns_default_when_no_file(tmp_path, monkeypatch):
    import train as train_module
    monkeypatch.setattr(train_module, "BEST_PARAMS_PATH", tmp_path / "nonexistent.json")
    params = load_clf_params()
    assert params == DEFAULT_CLF_PARAMS


def test_load_clf_params_reads_from_file(tmp_path, monkeypatch):
    import train as train_module
    best = {"params": {"n_estimators": 500, "learning_rate": 0.01}, "macro_f1": 0.91}
    p = tmp_path / "best_params.json"
    p.write_text(json.dumps(best))
    monkeypatch.setattr(train_module, "BEST_PARAMS_PATH", p)
    params = load_clf_params()
    assert params == best["params"]


def test_pipeline_fits_on_small_data():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(300, 5)), columns=[f"f{i}" for i in range(5)])
    y = pd.Series((rng.random(300) > 0.85).astype(int))
    pipe = build_pipeline()
    pipe.fit(X, y)
    preds = pipe.predict(X)
    assert len(preds) == len(X)

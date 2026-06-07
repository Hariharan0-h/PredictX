"""Tests for SHAP explainability (src/explain.py)."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd
import pytest
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier
from sklearn.preprocessing import StandardScaler

from explain import extract_lgbm_and_transform


@pytest.fixture
def tiny_pipeline():
    """Fit a minimal pipeline on synthetic binary data."""
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(200, 5)), columns=[f"f{i}" for i in range(5)])
    y = pd.Series((rng.random(200) > 0.85).astype(int))

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("smote", SMOTE(random_state=0)),
        ("clf", LGBMClassifier(n_estimators=10, random_state=0, verbose=-1)),
    ])
    pipe.fit(X, y)
    return pipe, X, y


def test_extract_transform_shape(tiny_pipeline):
    pipe, X, _ = tiny_pipeline
    X_t = extract_lgbm_and_transform(pipe, X)
    # Transformed matrix must have same number of columns as input
    assert X_t.shape[1] == X.shape[1]


def test_extract_transform_no_smote_inflation(tiny_pipeline):
    """Inference transform must NOT inflate row count (SMOTE is train-only)."""
    pipe, X, _ = tiny_pipeline
    X_t = extract_lgbm_and_transform(pipe, X)
    assert X_t.shape[0] == X.shape[0]


def test_shap_values_shape(tiny_pipeline):
    import shap
    pipe, X, _ = tiny_pipeline
    clf = pipe.named_steps["clf"]
    X_t = extract_lgbm_and_transform(pipe, X)
    if not isinstance(X_t, pd.DataFrame):
        X_t = pd.DataFrame(X_t, columns=X.columns)

    explainer = shap.TreeExplainer(clf)
    sv = explainer.shap_values(X_t)
    if isinstance(sv, list):
        sv = sv[1]

    assert sv.shape == X_t.shape, (
        f"SHAP values shape {sv.shape} must match input shape {X_t.shape}"
    )

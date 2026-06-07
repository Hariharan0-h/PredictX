"""
Tests that SMOTE is applied only within training folds (no leakage).
Verifies the pipeline uses imblearn.Pipeline, not sklearn.Pipeline.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from train import build_pipeline


def test_pipeline_is_imblearn():
    """Pipeline must be imblearn.Pipeline to handle SMOTE inside CV correctly."""
    pipe = build_pipeline()
    assert isinstance(pipe, ImbPipeline), (
        "Pipeline must be imblearn.Pipeline, not sklearn.Pipeline. "
        "Only imblearn.Pipeline correctly applies SMOTE within CV folds."
    )


def test_smote_is_in_pipeline():
    pipe = build_pipeline()
    step_types = [type(step).__name__ for _, step in pipe.steps]
    assert "SMOTE" in step_types, "SMOTE must be a step in the pipeline"


def test_smote_before_classifier():
    pipe = build_pipeline()
    step_names = [name for name, _ in pipe.steps]
    smote_idx = step_names.index("smote")
    clf_idx = step_names.index("clf")
    assert smote_idx < clf_idx, "SMOTE must come before the classifier in the pipeline"

"""Tests for external context simulation and merge (src/context.py)."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd
import pytest

from context import simulate_external_context


def test_context_shape():
    ctx = simulate_external_context(100)
    assert ctx.shape == (100, 2)


def test_context_columns():
    ctx = simulate_external_context(50)
    assert set(ctx.columns) == {"ambient_temp_ext", "factory_load"}


def test_factory_load_bounds():
    ctx = simulate_external_context(10_000, seed=42)
    assert ctx["factory_load"].min() >= 0.4
    assert ctx["factory_load"].max() <= 1.0


def test_ambient_temp_distribution():
    ctx = simulate_external_context(10_000, seed=42)
    mean = ctx["ambient_temp_ext"].mean()
    assert abs(mean - 20.0) < 0.5, f"Expected mean ~20, got {mean:.2f}"


def test_reproducibility():
    ctx1 = simulate_external_context(100, seed=0)
    ctx2 = simulate_external_context(100, seed=0)
    pd.testing.assert_frame_equal(ctx1, ctx2)

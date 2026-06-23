"""Tests for src/report.py — model report generator."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from report import build_report, _load


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def test_build_report_returns_string():
    report = build_report()
    assert isinstance(report, str)


def test_report_has_title():
    report = build_report()
    assert "# PredictX" in report


def test_report_has_all_section_headers():
    report = build_report()
    for section in [
        "## Cross-Validation Results",
        "## Evaluation on Noisy Test Set",
        "## Tuned Hyperparameters",
        "## Top-10 Features by SHAP Importance",
    ]:
        assert section in report


def test_report_graceful_when_no_files(monkeypatch, tmp_path):
    """All sections show 'not found' messages when result files are absent."""
    import report as report_module
    for attr in ["CV_SCORES_PATH", "EVAL_METRICS_PATH", "BEST_PARAMS_PATH",
                 "SHAP_TOP10_PATH", "DRIFT_REPORT_PATH"]:
        monkeypatch.setattr(report_module, attr, tmp_path / f"nonexistent_{attr}.json")
    r = build_report()
    assert "not found" in r


def test_report_includes_cv_fold_scores(monkeypatch, tmp_path):
    import report as report_module
    p = tmp_path / "cv_scores.json"
    write_json(p, {"per_fold_f1": [0.87, 0.88, 0.86, 0.89, 0.90],
                   "mean_macro_f1": 0.88, "std_macro_f1": 0.014})
    monkeypatch.setattr(report_module, "CV_SCORES_PATH", p)
    monkeypatch.setattr(report_module, "EVAL_METRICS_PATH",  tmp_path / "x.json")
    monkeypatch.setattr(report_module, "BEST_PARAMS_PATH",   tmp_path / "x2.json")
    monkeypatch.setattr(report_module, "SHAP_TOP10_PATH",    tmp_path / "x3.json")
    monkeypatch.setattr(report_module, "DRIFT_REPORT_PATH",  tmp_path / "x4.json")
    r = build_report()
    assert "0.8800" in r
    assert "0.014" in r


def test_report_includes_tuned_threshold(monkeypatch, tmp_path):
    import report as report_module
    p = tmp_path / "eval_metrics.json"
    write_json(p, {"default_threshold_macro_f1": 0.81,
                   "tuned_threshold": 0.37,
                   "tuned_threshold_macro_f1": 0.88,
                   "noise_scale": 0.1})
    monkeypatch.setattr(report_module, "CV_SCORES_PATH",     tmp_path / "x.json")
    monkeypatch.setattr(report_module, "EVAL_METRICS_PATH",  p)
    monkeypatch.setattr(report_module, "BEST_PARAMS_PATH",   tmp_path / "x2.json")
    monkeypatch.setattr(report_module, "SHAP_TOP10_PATH",    tmp_path / "x3.json")
    monkeypatch.setattr(report_module, "DRIFT_REPORT_PATH",  tmp_path / "x4.json")
    r = build_report()
    assert "0.3700" in r


def test_report_includes_drift_section_when_file_present(monkeypatch, tmp_path):
    import report as report_module
    p = tmp_path / "drift_report.json"
    write_json(p, {
        "alpha": 0.05,
        "drift_detected": True,
        "drifted_features": ["Torque [Nm]"],
        "features": {
            "Torque [Nm]": {"ks_statistic": 0.31, "p_value": 0.001, "drifted": True}
        }
    })
    for attr, name in [("CV_SCORES_PATH", "a"), ("EVAL_METRICS_PATH", "b"),
                       ("BEST_PARAMS_PATH", "c"), ("SHAP_TOP10_PATH", "d")]:
        monkeypatch.setattr(report_module, attr, tmp_path / f"{name}.json")
    monkeypatch.setattr(report_module, "DRIFT_REPORT_PATH", p)
    r = build_report()
    assert "## Feature Drift Report" in r
    assert "Drift detected" in r


def test_load_returns_none_for_missing_file(tmp_path):
    result = _load(tmp_path / "nonexistent.json")
    assert result is None


def test_load_returns_dict_for_existing_file(tmp_path):
    p = tmp_path / "data.json"
    write_json(p, {"key": "value"})
    result = _load(p)
    assert result == {"key": "value"}

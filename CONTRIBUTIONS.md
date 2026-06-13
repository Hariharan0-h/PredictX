# Contributions

## Team

| Name | Role |
|---|---|
| Hariharan | ML Engineer (Member 1) |
| Vanshika | ML Engineer (Member 2) |

---

## Hariharan — Task Ownership

| Task | File | Status |
|---|---|---|
| Task 1.1 — Data Ingestion | `src/ingest.py` | ✅ Done |
| Task 1.2 — Rolling Features | `src/features.py` | ✅ Done |
| Task 1.3 — External Context + Ablation | `src/context.py` | ✅ Done |
| Task 1.4 — SMOTE + LightGBM Pipeline | `src/train.py` | ✅ Done |
| Task 1.5 — Noise Injection + Threshold Tuning | `src/evaluate.py` | ✅ Done |
| SHAP Explainability | `src/explain.py` | ✅ Done |
| Hyperparameter Tuning | `src/tune.py` | ✅ Done |
| EDA Notebook | `notebooks/01_eda.ipynb` | ✅ Done |
| Test Suite | `tests/` | ✅ Done |

---

## Hariharan — Week-by-Week Log

**Project:** Contextual Predictive Maintenance (IoT Edge AI)
**Company:** Infotact Solutions & Co.

### Week 1: Pipeline Foundation
- Implemented full data ingestion pipeline (`src/ingest.py`) with AI4I 2020 dataset validation and failure-rate assertion.
- Engineered rolling window features (`src/features.py`) — mean/std/var with window=10 on 5 sensor columns.
- Simulated external context (`src/context.py`) — ambient_temp_ext, factory_load; ran ablation study proving fused features lift Macro F1.
- Built SMOTE-inside-CV LightGBM pipeline (`src/train.py`) using `imblearn.Pipeline` to prevent data leakage; asserts Macro F1 ≥ 0.85.
- Implemented noise injection + PR-curve threshold tuning (`src/evaluate.py`).
- Added SHAP TreeExplainer explainability (`src/explain.py`) — bar + beeswarm plots, top-10 JSON.
- Wrote Optuna TPE hyperparameter sweep (`src/tune.py`) — 50 trials, saves `results/best_params.json` for auto-load in train.py.

### Week 2: Observability & Test Coverage
- Wired `custom_logging` structured logger into all 7 src files — replaced all `print()` calls with `logger.info()`.
- Authored `tests/test_logging.py` — 6 tests covering import, Logger type, name, file creation, dir-not-file regression, message write.
- Added `tests/test_evaluate.py` (7 tests) and `tests/test_tune.py` (4 tests) covering noise injection, threshold validation, and Optuna objective.
- Added `logs/` to `.gitignore`; annotated `requirements.txt` clarifying `custom_logging` uses stdlib only.
- Added root `conftest.py` and `tests/conftest.py` to fix pytest path resolution without removing `tests/__init__.py`.

---

## Vanshika — Week-by-Week Log

**Project:** Contextual Predictive Maintenance (IoT Edge AI)
**Company:** Infotact Solutions & Co.

### Week 1: Environment Setup & Telemetry Ingestion
- **Local Setup:** Successfully cloned the repository to local environment using Git.
- **Environment Verification:** Configured VS Code and verified repository structure (`data`, `notebooks`, `src`, `tests`).
- **Dataset Study:** Initiated documentation review for the AI4I Predictive Maintenance Dataset to understand rolling statistical feature requirements.
- **Custom Logging:** Added `custom_logging/` module for structured pipeline logging.
- **Current Status:** Workspace fully operational and prepared for time-series signal processing.

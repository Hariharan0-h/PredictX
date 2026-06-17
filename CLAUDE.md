# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Contextual Predictive Maintenance system (Infotact DS/ML Internship — Project 1). Combines IoT sensor data (AI4I 2020 dataset) with simulated external context to forecast machine failures using LightGBM + SMOTE. **Primary KPI: Macro F1 ≥ 0.85 under synthetic sensor noise.**

## Commands

```bash
# Environment setup
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Run the full pipeline in order
python src/ingest.py          # Task 1.1 — download & validate data
python src/features.py        # Task 1.2 — rolling window features → data/features_internal.parquet
python src/context.py         # Task 1.3 — simulate & merge external context → data/features_fused.parquet
python src/train.py           # Task 1.4 — SMOTE-inside-CV LightGBM; asserts Macro F1 ≥ 0.85
python src/evaluate.py        # Task 1.5 — noise injection, PR curve, threshold tuning

# SHAP explainability
python src/explain.py

# Hyperparameter tuning (run before train.py to inject best params)
python src/tune.py --trials 50

# Inference on new data
python src/predict.py --input path/to/sensors.csv

# Feature drift detection
python src/drift.py --train data/features_fused.parquet --new path/to/new.csv

# Generate model performance report
python src/report.py

# Run a single test
pytest tests/test_features.py -v
pytest tests/ -k "test_smote_leakage" -v

# Strip notebook outputs before committing
nbstripout notebooks/*.ipynb
```

## Architecture

```
.
├── data/                  # gitignored — never commit CSVs or parquets
│   ├── features_internal.parquet
│   └── features_fused.parquet
├── models/                # gitignored — never commit .pt/.h5
├── notebooks/             # EDA only; outputs stripped on commit via nbstripout
├── outputs/               # plots: pr_curve_noisy.png, etc.
├── results/               # JSON metric logs
├── src/
│   ├── ingest.py          # Downloads AI4I dataset; asserts failure rate < 5%
│   ├── features.py        # Rolling window (mean/std/var, window=10) on 5 sensor cols
│   ├── context.py         # Simulates ambient_temp_ext + factory_load; ablation study
│   ├── train.py           # Pipeline: StandardScaler → SMOTE → LightGBMClassifier
│   ├── evaluate.py        # Noise injection (σ=0.1×feat_std), PR curve, threshold sweep
│   ├── explain.py         # SHAP TreeExplainer — bar + beeswarm plots, top-10 JSON
│   ├── tune.py            # Optuna TPE sweep (50 trials) → results/best_params.json
│   ├── predict.py         # Inference: loads pipeline.pkl + tuned threshold, scores CSV
│   ├── drift.py           # KS-test feature drift detection vs training baseline
│   └── report.py          # Collates all results/ JSON into results/model_report.md
├── tests/
├── custom_logging/        # Stdlib structured logger — no pip install needed
├── exception/             # CustomException: captures script name + line number on failure
└── requirements.txt
```

**Pipeline data flow:** raw CSV → `features_internal.parquet` → `features_fused.parquet` → trained pipeline → noisy test evaluation.

## Critical Constraints

**SMOTE leakage (non-negotiable):** SMOTE must live inside a `sklearn.Pipeline` and only fit on training folds. Never apply SMOTE before `StratifiedKFold` splitting. Violation = data leakage.

```python
# Correct pattern
from imblearn.pipeline import Pipeline  # use imblearn's Pipeline, not sklearn's
pipe = Pipeline([('scaler', StandardScaler()), ('smote', SMOTE()), ('clf', LGBMClassifier())])
cross_val_score(pipe, X, y, cv=StratifiedKFold(5), scoring='f1_macro')
```

**Decision threshold:** Tune on validation PR curve — do not use default 0.5.

**Ablation study (Task 1.3):** Must report Macro F1 delta between internal-only vs. fused features to prove external context adds lift.

## Commit & Git Rules

- **Format:** `feat|model|data|fix: <description> (fixes #<issue>)`
- **4 consecutive weeks of commits required** — single-push submissions disqualify.
- **Never push:** `data/`, `models/`, `*.csv`, `*.parquet`, `*.pt`, `*.h5`
- GitHub Kanban: Issues move To Do → In Progress → Done per task.

## Key Implementation Details

- **Dataset:** AI4I 2020 Predictive Maintenance (UCI ML Repository), target column: `Machine failure`, expected failure rate < 5%.
- **Sensor columns for rolling features:** `Air temperature`, `Process temperature`, `Rotational speed`, `Torque`, `Tool wear` — all get `_roll_mean`, `_roll_std`, `_roll_var` with `window=10`.
- **External context simulation:** `ambient_temp_ext ~ N(20, 5)`, `factory_load ~ Uniform(0.4, 1.0)`, merged by index.
- **Noise injection (Task 1.5):** Gaussian noise with σ = 0.1 × per-feature std applied to all sensor columns on the held-out test set only.
- **CV config:** `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`.

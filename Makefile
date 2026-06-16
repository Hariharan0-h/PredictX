.PHONY: pipeline tune predict drift report test coverage clean

# Run the full ML pipeline end-to-end
pipeline:
	python src/ingest.py
	python src/features.py
	python src/context.py
	python src/train.py
	python src/evaluate.py
	python src/explain.py

# Run hyperparameter tuning then retrain with best params
tune:
	python src/tune.py --trials 50
	python src/train.py

# Run inference on new data (override INPUT and THRESHOLD as needed)
# Usage: make predict INPUT=path/to/sensors.csv
#        make predict INPUT=path/to/sensors.csv THRESHOLD=0.35
INPUT ?= outputs/predictions.csv
THRESHOLD ?=
predict:
	python src/predict.py --input $(INPUT) $(if $(THRESHOLD),--threshold $(THRESHOLD),)

# Run feature drift detection
# Usage: make drift NEW=path/to/new_sensors.csv
NEW ?=
drift:
	python src/drift.py --train data/features_fused.parquet --new $(NEW)

# Generate Markdown model performance report from results/ JSON files
report:
	python src/report.py

# Run the full test suite
test:
	pytest tests/ -v

# Run tests with coverage report
coverage:
	pytest tests/ -v --cov=src --cov-report=term-missing --cov-report=html:outputs/coverage

# Remove generated data/model artifacts (keeps .gitkeep files)
clean:
	find data/ -name "*.parquet" -delete
	find data/ -name "*.csv" -delete
	find models/ -name "*.pkl" -delete
	find results/ -name "*.json" -delete
	find outputs/ -name "*.png" -delete
	@echo "Cleaned generated artifacts."

.PHONY: pipeline tune test clean

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

# Run the full test suite
test:
	pytest tests/ -v

# Remove generated data/model artifacts (keeps .gitkeep files)
clean:
	find data/ -name "*.parquet" -delete
	find data/ -name "*.csv" -delete
	find models/ -name "*.pkl" -delete
	find results/ -name "*.json" -delete
	find outputs/ -name "*.png" -delete
	@echo "Cleaned generated artifacts."

#!/bin/bash

set -x

# 1. Create the virtual environment
python -m venv .venv

# 2. Activate it
. .venv/bin/activate

# 3. Installs core dependencies (XGBoost is skipped because it's in an optional group)
poetry install

# 4. Extract version and inject without deps
XGB_VERSION=$(grep "xgboost =" pyproject.toml | sed 's/.*"\(.*\)".*/\1/')
poetry run pip install "xgboost==$XGB_VERSION" --no-deps

# Quick test
# poetry run python src/preprocess.py --data.path_in data_in --data.path_out data_out
# poetry run python src/extract_features.py --data.dataset savee --data.features=all --data.path_in ./data/interim/SAVEE_VAD --data.path_out ./data/features-333/
# poetry run python src/run_experiment.py --model xgb_classifier --data ravdess --data.path data/features/RAVDESS/all_labeled.tsv --report all
# poetry run python src/batch_runner.py batch.max_runs=2
# poetry run python -m unittest discover -s tests -v

# poetry run python src/run_experiment.py --data ravdess --model xgb_classifier --report basic --experiment.scaler robust --report.data_debug --features top20
# poetry run python src/run_experiment.py --data savee --model xgb_classifier --report basic --experiment.scaler robust --report.data_debug --features top20
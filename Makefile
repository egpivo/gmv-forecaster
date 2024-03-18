SHELL := /bin/bash

.PHONY: clean install train forecast

clean: clean-pyc clean-build

clean-pyc:
	@find . -type f -name '*.py[co]' -delete -o -type d -name __pycache__ -delete

clean-build:
	@rm -fr build/ dist/ .eggs/
	@find . -name '*.egg-info' -o -name '*.egg' -exec rm -fr {} +

install:
	@$(SHELL) envs/conda/build_conda_env.sh

train: install
	@eval "$$(conda shell.bash hook)" && \
	conda activate forecaster && \
	python forecaster/run_training.py \
		--user_data_path "data/sources/users.csv" \
		--transaction_data_path "data/sources/transactions.csv" \
		--store_data_path "data/sources/stores.csv"

forecast: install
	@eval "$$(conda shell.bash hook)" && \
	conda activate forecaster && \
	python forecaster/run_forecasting.py \
		--user_data_path "data/sources/users.csv" \
		--transaction_data_path "data/sources/transactions.csv" \
		--store_data_path "data/sources/stores.csv"

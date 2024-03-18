SHELL := /bin/bash

.PHONY: install train forecast

install:
	@$(SHELL) envs/conda/build_conda_env.sh

train: install
	@eval "$$(conda shell.bash hook)" && \
	conda activate forecaster && \
	python forecaster/run_training.py \
		--user_data_path "data/users.csv" \
		--transaction_data_path "data/transactions.csv" \
		--store_data_path "data/stores.csv"

forecast: install
	@eval "$$(conda shell.bash hook)" && \
	conda activate forecaster && \
	python forecaster/run_forecasting.py \
		--user_data_path "data/users.csv" \
		--transaction_data_path "data/transactions.csv" \
		--store_data_path "data/stores.csv"

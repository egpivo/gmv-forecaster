SHELL := /bin/bash

.PHONY: install run

install:
	@$(SHELL) envs/conda/build_conda_env.sh

run: install
	@eval "$$(conda shell.bash hook)" && \
	conda activate forecaster && \
	python forecaster/main.py \
		--user_data_path "data/users.csv" \
		--transaction_data_path "data/transactions.csv" \
		--store_data_path "data/stores.csv" \
		--end_date "2021-12-01"

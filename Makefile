SHELL := /bin/bash

.PHONY: install

install:
	@$(SHELL) envs/conda/build_conda_env.sh

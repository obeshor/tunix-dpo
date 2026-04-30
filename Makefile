# Tunix DPO — Makefile for common dev tasks
.PHONY: install install-dev test test-unit test-integration lint type fmt clean curate train eval export serve docker-build

PYTHON ?= python3.11

install:
	pip install -e .

install-dev:
	pip install -e '.[all]'
	pre-commit install

test:
	pytest

test-unit:
	pytest tests/unit -v

test-integration:
	pytest tests/integration -v

lint:
	ruff check src tests

fmt:
	ruff format src tests

type:
	mypy src

clean:
	rm -rf build/ dist/ *.egg-info/ .pytest_cache/ .mypy_cache/ .ruff_cache/
	find . -type d -name __pycache__ -exec rm -rf {} +

curate:
	tunix-curate --format dpo --output_dir ./data

train:
	tunix-train --config configs/dpo_v5e.yaml

eval:
	tunix-eval --base_model google/gemma-3-1b-it \
	           --tuned_model ./checkpoints/dpo_v5e_run/final \
	           --output_dir ./benchmark_results

export:
	tunix-export --checkpoint_dir ./checkpoints/dpo_v5e_run/final \
	             --base_model google/gemma-3-1b-it \
	             --output_dir ./exports/tunix_dpo_gemma_3_1b_it

serve:
	tunix-serve --model ./exports/tunix_dpo_gemma_3_1b_it --port 8000

docker-build:
	docker build -t tunix-dpo:latest .

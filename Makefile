# ─────────────────────────────────────────────────────────────────────────────
# Makefile — tunix_dpo project automation
# ─────────────────────────────────────────────────────────────────────────────
# Usage:
#   make install       install all dependencies (editable mode)
#   make test          run unit + integration tests
#   make test-unit     run unit tests only
#   make lint          ruff check + mypy
#   make fmt           ruff format (auto-fix)
#   make curate        Phase 1 — curate HH-RLHF data
#   make train         Phase 2 — launch DPO training
#   make eval          Phase 3 — run TruthfulQA + ToxiGen benchmarks
#   make export        Phase 4 — export weights to safetensors
#   make serve         Phase 4 — start vLLM inference server
#   make clean         remove build artefacts
# ─────────────────────────────────────────────────────────────────────────────

.PHONY: install test test-unit test-integration lint fmt \
        curate train eval export serve clean

PYTHON   := python
SRC      := src/tunix_dpo
TESTS    := tests

# ── Setup ─────────────────────────────────────────────────────────────────────

install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -e ".[all]"
	pre-commit install

# ── Tests ─────────────────────────────────────────────────────────────────────

test:
	$(PYTHON) -m pytest $(TESTS) -v --tb=short

test-unit:
	$(PYTHON) -m pytest $(TESTS)/unit -v --tb=short

test-integration:
	$(PYTHON) -m pytest $(TESTS)/integration -v --tb=short

# ── Code quality ──────────────────────────────────────────────────────────────

lint:
	$(PYTHON) -m ruff check $(SRC) $(TESTS)
	$(PYTHON) -m mypy $(SRC)

fmt:
	$(PYTHON) -m ruff format $(SRC) $(TESTS)
	$(PYTHON) -m ruff check --fix $(SRC) $(TESTS)

# ── Phase 1: Data curation ────────────────────────────────────────────────────

DATA_DIR     ?= ./data
DATA_FORMAT  ?= dpo
DATA_SUBSETS ?= helpful-base harmless-base

curate:
	tunix-curate \
		--format      $(DATA_FORMAT) \
		--subsets     $(DATA_SUBSETS) \
		--output_dir  $(DATA_DIR)

validate:
	$(PYTHON) -c "\
from tunix_dpo.data.curate import write_jsonl; \
import json, pathlib; \
path = pathlib.Path('$(DATA_DIR)/dpo/train.jsonl'); \
rows = [json.loads(l) for l in path.read_text().splitlines()]; \
assert all({'prompt','chosen','rejected'} <= r.keys() for r in rows); \
print(f'OK — {len(rows)} valid DPO records')"

# ── Phase 2: Training ─────────────────────────────────────────────────────────

CONFIG      ?= configs/dpo_v5e.yaml

train:
	tunix-train --config $(CONFIG)

plot:
	$(PYTHON) -m tunix_dpo.training.logger  # placeholder; run plot_trajectories.py

# ── Phase 3: Evaluation ───────────────────────────────────────────────────────

BASE_MODEL   ?= google/gemma-2b
TUNED_MODEL  ?= ./checkpoints/dpo_v5e_run/final
EVAL_OUT     ?= ./benchmark_results
MAX_Q        ?= 817

eval:
	tunix-eval \
		--base_model    $(BASE_MODEL) \
		--tuned_model   $(TUNED_MODEL) \
		--output_dir    $(EVAL_OUT) \
		--max_questions $(MAX_Q)

eval-quick:
	tunix-eval \
		--base_model      $(BASE_MODEL) \
		--tuned_model     $(TUNED_MODEL) \
		--output_dir      $(EVAL_OUT) \
		--max_questions   50 \
		--max_per_group   20

# ── Phase 4: Export & serving ─────────────────────────────────────────────────

EXPORT_DIR   ?= ./exports/tunix_dpo_gemma2b
SERVER_PORT  ?= 8000

export:
	tunix-export \
		--checkpoint_dir $(TUNED_MODEL) \
		--base_model     $(BASE_MODEL) \
		--output_dir     $(EXPORT_DIR)

export-litert:
	tunix-export \
		--checkpoint_dir $(TUNED_MODEL) \
		--base_model     $(BASE_MODEL) \
		--output_dir     $(EXPORT_DIR) \
		--export_litert \
		--litert_quant   int8

serve:
	tunix-serve \
		--model $(EXPORT_DIR) \
		--port  $(SERVER_PORT)

serve-quantized:
	tunix-serve \
		--model         $(EXPORT_DIR)_gptq \
		--quantization  gptq \
		--port          $(SERVER_PORT)

# ── Docker ────────────────────────────────────────────────────────────────────

IMAGE_NAME   ?= tunix-dpo-serve
GCP_PROJECT  ?= your-gcp-project

docker-build:
	docker build -t $(IMAGE_NAME) .

docker-run:
	docker run --gpus all -p $(SERVER_PORT):$(SERVER_PORT) \
		-v $(EXPORT_DIR):/model \
		$(IMAGE_NAME)

docker-push:
	docker tag $(IMAGE_NAME) gcr.io/$(GCP_PROJECT)/$(IMAGE_NAME)
	docker push gcr.io/$(GCP_PROJECT)/$(IMAGE_NAME)

# ── Cleanup ───────────────────────────────────────────────────────────────────

clean:
	rm -rf dist/ build/ .eggs/ *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc"     -delete
	find . -type f -name ".coverage" -delete
	rm -rf .mypy_cache .ruff_cache .pytest_cache htmlcov/

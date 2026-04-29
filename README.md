# tunix-dpo

End-to-end DPO alignment pipeline: HH-RLHF curation → TPU training → TruthfulQA/ToxiGen evaluation → vLLM serving.

[![Tests](https://img.shields.io/badge/tests-pytest-blue)](tests/)
[![Python](https://img.shields.io/badge/python-3.11-blue)](pyproject.toml)
[![License](https://img.shields.io/badge/license-Apache--2.0-green)](LICENSE)

---

## Results

| Benchmark | Metric | Base | DPO-tuned | Δ |
|-----------|--------|------|-----------|---|
| TruthfulQA | Binary accuracy ★ | 0.512 | 0.614 | **+19.9%** |
| TruthfulQA | Calibration error ↓ | 0.341 | 0.274 | −19.6% |
| ToxiGen | Gen. toxicity rate ↓ | 0.412 | 0.218 | **−47.1%** |
| ToxiGen | Discrimination accuracy | 0.631 | 0.748 | +18.5% |
| Serving | Peak throughput | — | **601 tok/s** | — |

All improvements are statistically significant (95% bootstrap CIs, non-overlapping).

---

## Project structure

```
tunix-dpo/
├── pyproject.toml              ← packaging, deps, entry points, ruff/mypy config
├── Makefile                    ← all common tasks as `make <target>`
├── Dockerfile                  ← GPU inference container (Cloud Run ready)
├── .pre-commit-config.yaml     ← ruff + mypy + trailing-whitespace hooks
├── .gitignore
│
├── configs/
│   └── dpo_v5e.yaml            ← all training hyperparameters
│
├── scripts/
│   └── tpu_provision.sh        ← provision GCP v5e-8 TPU VM
│
├── src/
│   └── tunix_dpo/
│       ├── data/
│       │   ├── parser.py       ← parse_dialogue(), is_valid_pair()  [pure]
│       │   ├── formatter.py    ← format_dpo(), format_rm()          [pure]
│       │   └── curate.py       ← CLI: tunix-curate
│       │
│       ├── training/
│       │   ├── config.py       ← dataclasses only, no logic
│       │   ├── losses.py       ← dpo_loss(), sft_loss()             [pure JAX]
│       │   ├── dataset.py      ← HHRLHFDataset + DataLoader
│       │   ├── optimizer.py    ← make_optimizer()
│       │   ├── logger.py       ← TrajectoryLogger (TensorBoard + JSONL)
│       │   └── train.py        ← CLI: tunix-train
│       │
│       ├── evaluation/
│       │   ├── stats.py        ← bootstrap_ci(), cohens_h()         [pure]
│       │   ├── truthfulqa.py   ← MC1, MC2, Binary MC scoring        [pure]
│       │   ├── toxigen.py      ← ToxigenClassifier, gen/disc        [pure]
│       │   ├── compare.py      ← compare_truthfulqa/toxigen()       [pure]
│       │   └── runner.py       ← CLI: tunix-eval
│       │
│       └── serving/
│           ├── schemas.py      ← Pydantic models only, no FastAPI
│           ├── metrics.py      ← Metrics telemetry class
│           ├── engine.py       ← VLLMEngine async wrapper
│           ├── export.py       ← CLI: tunix-export  (Orbax → safetensors + LiteRT)
│           └── server.py       ← CLI: tunix-serve   (FastAPI app)
│
└── tests/
    ├── conftest.py             ← shared fixtures (no network, no GPU)
    ├── unit/
    │   ├── test_data.py        ← parser, formatter
    │   ├── test_losses.py      ← JAX loss functions
    │   ├── test_stats.py       ← statistical helpers
    │   ├── test_schemas.py     ← Pydantic models
    │   └── test_metrics.py     ← telemetry
    └── integration/
        ├── test_curate_pipeline.py  ← end-to-end data curation (mocked HF)
        └── test_compare.py          ← comparison logic with fixture data
```

---

## Quickstart

### Install

```bash
git clone https://github.com/your-org/tunix-dpo.git
cd tunix-dpo
make install        # pip install -e ".[all]" + pre-commit install
```

### Run the test suite

```bash
make test           # all tests
make test-unit      # fast, no network, no GPU required
```

### Phase 1 — Data curation

```bash
make curate DATA_FORMAT=dpo DATA_SUBSETS="helpful-base harmless-base"
# Output: ./data/dpo/train.jsonl (~157k rows), ./data/dpo/test.jsonl
```

Or directly:

```bash
tunix-curate \
    --format   dpo \
    --subsets  helpful-base harmless-base \
    --output_dir ./data
```

### Phase 2 — TPU training

```bash
# Provision GCP v5e-8 (one-time)
./scripts/tpu_provision.sh --project my-project --zone us-west4-a

# Copy files to TPU VM and train
gcloud compute tpus tpu-vm scp --recurse ./ tunix-dpo-v5e:~/ --zone=us-west4-a
gcloud compute tpus tpu-vm ssh tunix-dpo-v5e --zone=us-west4-a

# On the TPU VM:
tunix-train --config configs/dpo_v5e.yaml
```

Override any config key from the CLI:

```bash
tr
```

### Phase 3 — Evaluation

```bash
# Full benchmark (~2h on CPU, ~20min on GPU)
make eval BASE_MODEL=google/gemma-2b TUNED_MODEL=./checkpoints/dpo_v5e_run/final

# Quick smoke test (50 TruthfulQA + 20/group ToxiGen)
make eval-quick
```

### Phase 4 — Export & serve

```bash
# Export JAX weights → HuggingFace safetensors
make export TUNED_MODEL=./checkpoints/dpo_v5e_run/final EXPORT_DIR=./exports/tunix_dpo

# (Optional) also export LiteRT INT8 flatbuffer
make export-litert

# Launch vLLM inference server
make serve

# Test it
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Do vaccines cause autism?"}],"max_tokens":200}'
```

### Docker

```bash
make docker-build
make docker-run EXPORT_DIR=./exports/tunix_dpo

# Deploy to Cloud Run (NVIDIA L4)
make docker-push GCP_PROJECT=my-project
gcloud run deploy tunix-dpo \
  --image gcr.io/my-project/tunix-dpo-serve \
  --gpu 1 --gpu-type nvidia-l4 \
  --memory 32Gi --cpu 8 --port 8000 --allow-unauthenticated
```

---

## Entry points

After `pip install -e .`, the following commands are available:

| Command | Module | Description |
|---------|--------|-------------|
| `tunix-curate` | `tunix_dpo.data.curate` | Curate HH-RLHF → DPO / RM JSONL |
| `tunix-train` | `tunix_dpo.training.train` | DPO training on TPU |
| `tunix-eval` | `tunix_dpo.evaluation.runner` | TruthfulQA + ToxiGen evaluation |
| `tunix-export` | `tunix_dpo.serving.export` | Export JAX weights → safetensors + LiteRT |
| `tunix-serve` | `tunix_dpo.serving.server` | vLLM OpenAI-compatible API server |

---

## Design principles

Each module has exactly one reason to change:

- `parser.py`, `formatter.py`, `losses.py`, `stats.py`, `truthfulqa.py`, `toxigen.py`, `compare.py`, `schemas.py`, `metrics.py` — pure functions / data structures, no I/O, fully unit-testable.
- `curate.py`, `train.py`, `runner.py`, `export.py`, `server.py` — orchestration and I/O only; no business logic inline.
- `config.py` — dataclasses with defaults only, no methods.
- `engine.py` — one external dependency (vLLM) isolated behind a clean async interface.

---

## License

Apache 2.0. See [LICENSE](LICENSE).

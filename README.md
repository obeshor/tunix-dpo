# Tunix DPO

End-to-end DPO alignment pipeline: HH-RLHF curation → TPU training → TruthfulQA/ToxiGen evaluation → vLLM serving

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![JAX](https://img.shields.io/badge/JAX-0.4.30+-orange)](https://github.com/google/jax)
[![Gemma 3](https://img.shields.io/badge/model-gemma--3--1b--it-purple)](https://huggingface.co/google/gemma-3-1b-it)

Five command-line tools — `tunix-curate`, `tunix-train`, `tunix-eval`, `tunix-export`, `tunix-serve` —
covering the full lifecycle from raw HH-RLHF data to a live OpenAI-compatible API.

---

## Quick start

```bash
# 1. Install
pip install -e '.[all]'

# 2. Curate the HH-RLHF dataset
tunix-curate --format dpo --output_dir ./data

# 3. Train DPO on TPU
# Copy files to TPU VM and train
gcloud compute tpus tpu-vm scp --recurse ./ tunix-dpo-v5e:~/tunix-dpo --zone=us-west4-a
gcloud compute tpus tpu-vm ssh tunix-dpo-v5e --zone=us-west4-a
# On the TPU VM:
cd ~/tunix-dpo
pip install -e ".[training]"
pip install "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
# Verify  chips are visible
python -c "import jax; print(len(jax.devices()), 'TPU devices')"
tunix-train --config configs/dpo_v5e.yaml

# 4. Benchmark vs base
tunix-eval --base_model google/gemma-3-1b-it \
           --tuned_model ./checkpoints/dpo_v5e_run/final \
           --output_dir ./benchmark_results

# 5. Export for inference
tunix-export --checkpoint_dir ./checkpoints/dpo_v5e_run/final \
             --base_model google/gemma-3-1b-it \
             --output_dir ./exports/tunix_dpo_gemma_3_1b_it

# 6. Serve via OpenAI-compatible API
tunix-serve --model ./exports/tunix_dpo_gemma_3_1b_it --port 8000
```

## Project layout

```
tunix_dpo/
├── src/tunix_dpo/
│   ├── data/         # parser, formatter, curate CLI
│   ├── training/     # config, JAX losses, dataset, optimizer, train CLI
│   ├── evaluation/   # stats, TruthfulQA, ToxiGen, compare, runner CLI
│   └── serving/      # schemas, metrics, vLLM engine, export, server
├── tests/            # unit + integration
├── configs/          # YAML training configs
├── scripts/          # tpu_provision.sh
├── pyproject.toml
├── Makefile
├── Dockerfile
└── README.md
```

The Terraform infrastructure module lives in **`../infra/terraform/`** and provisions the TPU VM,
GCS bucket, service account, and IAM bindings.

## Five CLI commands

| Command         | Phase                | What it does                                              |
| --------------- | -------------------- | --------------------------------------------------------- |
| `tunix-curate`  | 1. Data              | Download HH-RLHF, parse, filter, write JSONL              |
| `tunix-train`   | 2. Training          | Run DPO on TPU v5e-8                                      |
| `tunix-eval`    | 3. Benchmarking      | TruthfulQA + ToxiGen vs base, with bootstrap CIs          |
| `tunix-export`  | 4. Conversion        | JAX/Orbax → HuggingFace safetensors (+ optional LiteRT)   |
| `tunix-serve`   | 4. Serving           | Start OpenAI-compatible vLLM API server                   |

## Model and zone choices

- **Model:** `google/gemma-3-1b-it` is the text-only 1B-parameter instruction-tuned variant of
  Gemma 3, with a 32K context window. Loads with `AutoModelForCausalLM` (or `Gemma3ForCausalLM`)
  in `transformers >= 4.50`.
- **Zone:** `us-west4-a` is one of the canonical Google Cloud zones for TPU v5e.
  See [Google's TPU v5e training docs](https://cloud.google.com/tpu/docs/v5e-training)
  which use this zone in their default examples.
- **TPU:** v5e-8 (`v5litepod-8` in `gcloud` syntax) — 8 chips, 192 GB HBM total. Holds Gemma 3 1B
  twice over (policy + frozen reference for DPO) with room to spare for activations.
- **Cost:** ~$12–16/hour on demand. A typical training run (157 K pairs, 1 epoch)
  finishes in 3–6 hours, so $50–100 per run.

## Architecture invariants

- **Pure-vs-impure separation.** Every package puts pure functions (`parser.py`, `formatter.py`,
  `losses.py`, `stats.py`, `truthfulqa.py`, `compare.py`, `schemas.py`, `metrics.py`) in their own
  module, separated from CLI/orchestration modules. Pure modules require **no** GPU, network, or
  optional dependencies for unit testing.
- **DPO prompt invariant.** The parser guarantees the prompt is byte-identical for chosen and
  rejected responses — DPO has no preference signal otherwise.
- **Frozen reference.** The training loop keeps the original model weights in `ref_params` and
  applies `jax.lax.stop_gradient` so gradients only flow through the policy.
- **Weight transposition.** The exporter automatically transposes Flax `[in, out]` Dense kernels
  to PyTorch `[out, in]`. Embedding tables and LayerNorm scales are left untransposed.

## Reproducibility

```bash
make test-unit            # ~2 s on CPU, no extras needed
make test-integration     # ~10 s, mocks HuggingFace
make lint type            # ruff + mypy
```
## Docker

```bash
make docker-build
make docker-run EXPORT_DIR=./exports/tunix_dpo

# Deploy to Cloud Run (NVIDIA L4)
make docker-push GCP_PROJECT=my-project
gcloud run deploy tunix-dpo \
  --image gcr.io/my-project/tunix-dpo-serve \
  --gpu 1 --gpu-type nvidia-l4 \
  --memory 32Gi --cpu 8 --port 8000 --allow-unauthenticated          # ruff + mypy
```

## License

Apache-2.0

#!/usr/bin/env bash
# ─── TPU VM startup script ──────────────────────────────────────────────────
# Runs on first boot. Installs Python 3.11, JAX with TPU support, and the
# rest of the Tunix DPO training dependencies.
# ────────────────────────────────────────────────────────────────────────────

set -euxo pipefail

# Wait until APT is ready (TPU images sometimes lag at boot)
until sudo apt-get update -qq; do sleep 5; done

sudo apt-get install -y --no-install-recommends \
  python3.11 python3.11-venv python3.11-dev \
  python3-pip git curl ca-certificates

# Tunix DPO uses Python 3.11
sudo update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Set up a virtual environment for the training user
sudo -u "$(logname)" bash <<'EOF'
set -e
python3.11 -m venv ~/venv
source ~/venv/bin/activate
pip install --upgrade pip

# JAX with TPU support
pip install 'jax[tpu]' \
  -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

# Flax + Optax + Orbax for training
pip install flax optax orbax-checkpoint tensorboard

# Transformers (>= 4.50 required for Gemma 3)
pip install 'transformers>=4.50' datasets tokenizers huggingface-hub

# Project deps
pip install pyyaml click pydantic numpy scipy tqdm

echo "Startup complete. JAX devices:"
python -c 'import jax; print(jax.devices())'
EOF

# ─────────────────────────────────────────────────────────────────────────────
# This script runs once on the TPU VM immediately after it boots.
# It installs the Python environment required for Phase 2 training.
#
# To use this with Terraform, add a metadata_startup_script attribute to
# google_tpu_v2_vm.training (see main.tf comment below).
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

LOG="/var/log/tunix-startup.log"
exec > >(tee -a "$LOG") 2>&1

echo "=== Tunix DPO startup — $(date) ==="

# 1. Upgrade pip
pip install --upgrade pip -q

# 2. JAX with TPU support (must use the special release index)
pip install "jax[tpu]" \
  -f https://storage.googleapis.com/jax-releases/libtpu_releases.html -q

# 3. Training stack
pip install flax optax orbax-checkpoint tunix -q

# 4. Verify 8 TPU chips are visible
python3 -c "
import jax
devices = jax.devices()
assert len(devices) == 8, f'Expected 8 v5e chips, got {len(devices)}'
print(f'TPU check passed: {len(devices)} devices ({devices[0].device_kind})')
"

echo "=== Startup complete — $(date) ==="

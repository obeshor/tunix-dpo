#!/usr/bin/env bash
# ─── TPU v5e-8 provisioning helper for the Tunix DPO project ────────────────
# Provisions a TPU VM in us-west4-a using the gcloud CLI directly.
# Most users should prefer the Terraform module in infra/terraform/ — this
# script exists for quick interactive use.

set -euo pipefail

# ─── Required env vars ──────────────────────────────────────────────────────
: "${PROJECT_ID:?must set PROJECT_ID}"
: "${TPU_NAME:=tunix-dpo-v5e}"
: "${ZONE:=us-west4-a}"            # v5e zone
: "${REGION:=us-west4}"            # quota lives at the region level
: "${ACCELERATOR_TYPE:=v5litepod-8}"   # gcloud's name for v5e-8
: "${RUNTIME_VERSION:=v2-alpha-tpuv5-lite}"
: "${BUCKET_NAME:=${PROJECT_ID}-tunix-checkpoints}"

echo "════════════════════════════════════════════════════════════════════"
echo "  Tunix DPO :: TPU v5e-8 provisioner"
echo "════════════════════════════════════════════════════════════════════"
echo "  PROJECT_ID:        $PROJECT_ID"
echo "  TPU_NAME:          $TPU_NAME"
echo "  ZONE:              $ZONE"
echo "  REGION:            $REGION"
echo "  ACCELERATOR_TYPE:  $ACCELERATOR_TYPE"
echo "  RUNTIME_VERSION:   $RUNTIME_VERSION"
echo "  BUCKET_NAME:       gs://$BUCKET_NAME"
echo "════════════════════════════════════════════════════════════════════"

# 1. Enable required APIs
echo "[1/4] Enabling required APIs…"
gcloud services enable tpu.googleapis.com storage.googleapis.com \
                       compute.googleapis.com iam.googleapis.com \
                       --project="$PROJECT_ID"

# 2. Create the GCS bucket co-located with the TPU region
echo "[2/4] Creating GCS bucket gs://$BUCKET_NAME (region=$REGION)…"
if ! gsutil ls "gs://$BUCKET_NAME" >/dev/null 2>&1; then
  gsutil mb -p "$PROJECT_ID" -l "$REGION" -b on "gs://$BUCKET_NAME"
else
  echo "      (bucket already exists)"
fi

# 3. Provision the TPU VM
echo "[3/4] Creating TPU VM $TPU_NAME in $ZONE…"
gcloud compute tpus tpu-vm create "$TPU_NAME" \
  --project="$PROJECT_ID" \
  --zone="$ZONE" \
  --accelerator-type="$ACCELERATOR_TYPE" \
  --version="$RUNTIME_VERSION"

# 4. SSH and install dependencies
echo "[4/4] Installing dependencies on the TPU VM…"
gcloud compute tpus tpu-vm ssh "$TPU_NAME" \
  --project="$PROJECT_ID" \
  --zone="$ZONE" \
  --command="
    set -e
    sudo apt-get update -qq
    sudo apt-get install -y python3.11 python3.11-venv git
    python3.11 -m venv ~/venv
    source ~/venv/bin/activate
    pip install --upgrade pip
    pip install 'jax[tpu]' \
      -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
    pip install flax optax orbax-checkpoint
    pip install 'transformers>=4.50' datasets tokenizers
    pip install pyyaml click pydantic numpy scipy tensorboard
    echo 'TPU VM ready. JAX devices:'
    python -c 'import jax; print(jax.devices())'
  "

echo "════════════════════════════════════════════════════════════════════"
echo "  Done. Next steps:"
echo "    1. Clone tunix-dpo onto the VM and run:  pip install -e ."
echo "    2. tunix-curate --output_dir ./data"
echo "    3. tunix-train  --config configs/dpo_v5e.yaml \"
echo "                    infra.gcs_bucket=gs://$BUCKET_NAME"
echo "    4. ⚠️  TEAR DOWN when done:"
echo "       gcloud compute tpus tpu-vm delete $TPU_NAME \"
echo "         --project=$PROJECT_ID --zone=$ZONE --quiet"
echo "════════════════════════════════════════════════════════════════════"

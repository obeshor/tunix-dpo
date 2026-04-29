#!/usr/bin/env bash
# =============================================================================
# scripts/tpu_provision.sh — Provision a v5e-8 TPU pod slice on Google Cloud
# =============================================================================
# Usage:
#   chmod +x scripts/tpu_provision.sh
#   ./scripts/tpu_provision.sh --project my-project --zone us-central2-b
# =============================================================================
set -euo pipefail

PROJECT="${GCP_PROJECT:-your-gcp-project-id}"
ZONE="${TPU_ZONE:-us-west4-a}"
TPU_NAME="${TPU_NAME:-tunix-dpo-v5e}"
ACCELERATOR_TYPE="v5e-8"
RUNTIME_VERSION="tpu-vm-tf-2.16.1-pjrt"
DISK_SIZE="200"
PREEMPTIBLE="${PREEMPTIBLE:-false}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --zone)        ZONE="$2";       shift 2 ;;
    --project)     PROJECT="$2";    shift 2 ;;
    --name)        TPU_NAME="$2";   shift 2 ;;
    --preemptible) PREEMPTIBLE="true"; shift ;;
    *) echo "Unknown flag: $1"; exit 1 ;;
  esac
done

PREEMPTIBLE_FLAG=""
[[ "$PREEMPTIBLE" == "true" ]] && PREEMPTIBLE_FLAG="--preemptible"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  TPU Provisioning — Tunix DPO"
echo "  Project: $PROJECT  Zone: $ZONE  Name: $TPU_NAME"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# 1. Set project
gcloud config set project "$PROJECT"

# 2. Enable APIs
gcloud services enable tpu.googleapis.com storage.googleapis.com --quiet

# 3. Create GCS bucket for checkpoints
BUCKET="gs://${PROJECT}-tunix-checkpoints"
gsutil mb -l us-west4 "$BUCKET" 2>/dev/null || echo "Bucket exists: $BUCKET"

# 4. Create TPU VM
gcloud compute tpus tpu-vm create "$TPU_NAME" \
  --zone="$ZONE" \
  --accelerator-type="$ACCELERATOR_TYPE" \
  --version="$RUNTIME_VERSION" \
  --boot-disk-size="${DISK_SIZE}GB" \
  $PREEMPTIBLE_FLAG \
  --quiet

# 5. Bootstrap the Python environment on the VM
gcloud compute tpus tpu-vm ssh "$TPU_NAME" --zone="$ZONE" --command="
  set -e
  pip install --upgrade pip -q
  pip install 'jax[tpu]' -f https://storage.googleapis.com/jax-releases/libtpu_releases.html -q
  pip install flax optax orbax-checkpoint tunix -q
  pip install 'tunix-dpo[training,eval,serving]' -q
  python3 -c \"
import jax
devices = jax.devices()
assert len(devices) == 8, f'Expected 8 v5e chips, got {len(devices)}'
print(f'✓ {len(devices)} TPU devices ready ({devices[0].device_kind})')
\"
"

echo ""
echo "✓ TPU VM ready: $TPU_NAME"
echo ""
echo "Copy project files:"
echo "  gcloud compute tpus tpu-vm scp --recurse ./ $TPU_NAME:~/ --zone=$ZONE"
echo ""
echo "SSH in:"
echo "  gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE"
echo ""
echo "Tear down when done:"
echo "  gcloud compute tpus tpu-vm delete $TPU_NAME --zone=$ZONE --quiet"

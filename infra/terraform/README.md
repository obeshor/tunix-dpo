# Tunix DPO — Terraform module

Provisions the GCP infrastructure for Phase 2 (TPU training) of the Tunix DPO
project. Creates a v5e-4 TPU VM in `us-west4-a`, a GCS checkpoint bucket
co-located in `us-west4`, and the supporting service account + IAM bindings.

## Resources created

| Resource | Type | Purpose |
| --- | --- | --- |
| 4 × `google_project_service` | API | Enable TPU, Storage, Compute, IAM |
| `google_service_account.tpu_sa` | IAM | Identity for the TPU VM |
| 2 × `google_project_iam_member` | IAM | objectAdmin + logWriter for the SA |
| `google_storage_bucket.checkpoints` | Storage | Stores Orbax checkpoints |
| `google_tpu_v2_vm.training` | TPU | The v5e-8 training VM |

## Quick start

```bash
# 1. Authenticate gcloud
gcloud auth application-default login

# 2. Set up your variables
cd infra/terraform
cp environments/dev/terraform.tfvars.example environments/dev/terraform.tfvars
# edit terraform.tfvars: set your project_id

# 3. Init + plan + apply
terraform init
terraform plan  -var-file=environments/dev/terraform.tfvars
terraform apply -var-file=environments/dev/terraform.tfvars

# 4. SSH in (the exact command is in `terraform output ssh_command`)
$(terraform output -raw ssh_command)

# 5. ⚠️  TEAR DOWN once training is done
terraform destroy -var-file=environments/dev/terraform.tfvars
```

## Variable reference

| Variable | Default               | Notes                                |
| --- |-----------------------|--------------------------------------|
| `project_id` | *required*            | Your GCP project.                    |
| `region` | `us-west4`            | TPU quota lives at the region level. |
| `zone` | `us-west4-a`          | Must be a v5e zone (validated).      |
| `tpu_name` | `tunix-dpo-v5e`       | The VM name.                         |
| `accelerator_type` | `v5litepod-4`         | gcloud/TF naming for v5e-4.          |
| `runtime_version` | `v2-alpha-tpuv5-lite` | The v5e runtime.                     |
| `env` | `dev`                 | Label only.                          |

## Cost reminder

A `v5litepod-8` (TPU v5e-4) on demand costs roughly **$12–16/hour**. A typical
training run (1 epoch over 157K HH-RLHF pairs with Gemma 3 1B IT) takes
**3–6 hours**, so a complete run is **$50–100**.

The GCS bucket is set to `force_destroy = false`, so `terraform destroy` will
**not** delete it if it contains checkpoints — your trained weights are safe.
After verifying you no longer need them, delete the bucket separately:

```bash
gsutil -m rm -r gs://your-project-tunix-checkpoints
gsutil rb gs://your-project-tunix-checkpoints
```

## Why `us-west4-a`?

- It is one of Google's canonical TPU v5e zones (and the default in their
  [v5e training docs](https://cloud.google.com/tpu/docs/v5e-training)).
- Latency from `us-west4` GCS to the TPU VM is single-digit milliseconds —
  important when checkpointing every 500 steps.
- TPU quota is regional, so the bucket and the VM share the same quota
  pool (`us-west4`).

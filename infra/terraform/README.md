# infra/terraform — Tunix DPO Phase 2 Infrastructure

Terraform module that provisions everything Phase 2 needs on Google Cloud:

| Resource | Description |
|----------|-------------|
| `google_project_service` × 4 | Enables TPU, Storage, Compute, IAM APIs |
| `google_service_account` | Dedicated identity for the TPU VM |
| `google_project_iam_member` × 3 | Storage admin, TPU viewer, log writer |
| `google_storage_bucket` | Checkpoint + log storage, co-located with TPU |
| `google_storage_bucket_iam_member` | Grants the service account bucket access |
| `google_tpu_v2_vm` | v5e-8 TPU VM (8 chips, 192 GB HBM) |

---

## Prerequisites

### 1. Install Terraform

```bash
# macOS
brew install terraform

# Linux
sudo apt-get install -y gnupg software-properties-common
wget -O- https://apt.releases.hashicorp.com/gpg | gpg --dearmor | \
  sudo tee /usr/share/keyrings/hashicorp-archive-keyring.gpg
echo "deb [signed-by=/usr/share/keyrings/hashicorp-archive-keyring.gpg] \
  https://apt.releases.hashicorp.com $(lsb_release -cs) main" | \
  sudo tee /etc/apt/sources.list.d/hashicorp.list
sudo apt update && sudo apt-get install terraform

# Verify
terraform -version   # must be >= 1.6
```

### 2. Authenticate with Google Cloud

```bash
gcloud auth login
gcloud auth application-default login
gcloud config set project YOUR_PROJECT_ID
```

### 3. Request TPU v5e quota

TPU v5e requires a quota increase before it can be created.

Go to **Console → IAM & Admin → Quotas**, search for:
- Quota name: `TPUS-PER-PROJECT-per-zone`
- Region: `us-central1` (or your chosen region)

Request a limit of **8** chips. Approval typically takes a few minutes to
a few hours for GDE sprint accounts.

---

## File structure

```
infra/terraform/
├── main.tf                        ← providers, APIs, SA, bucket, TPU VM
├── variables.tf                   ← all input variables with defaults
├── outputs.tf                     ← SSH command, bucket URL, etc.
├── .gitignore                     ← excludes .tfstate and *.tfvars
├── scripts/
│   └── startup.sh                 ← JAX bootstrap script for the VM
└── environments/
    └── dev/
        └── terraform.tfvars       ← YOUR VALUES (not committed to git)
```

---

## Step-by-step usage

### Step 1 — Fill in your values

Edit `environments/dev/terraform.tfvars`:

```hcl
project_id = "your-actual-project-id"
zone       = "us-central1-a"
region     = "us-central1"
```

### Step 2 — Initialise Terraform

Run once per machine (downloads the Google provider plugin):

```bash
cd infra/terraform
terraform init
```

You should see:
```
Terraform has been successfully initialized!
```

### Step 3 — Preview the plan

See exactly what will be created before touching anything:

```bash
terraform plan -var-file=environments/dev/terraform.tfvars
```

Expected output — 10 resources to add:

```
Plan: 10 to add, 0 to change, 0 to destroy.

  + google_project_service.tpu
  + google_project_service.storage
  + google_project_service.compute
  + google_project_service.iam
  + google_service_account.tpu_sa
  + google_project_iam_member.tpu_sa_storage
  + google_project_iam_member.tpu_sa_tpu_viewer
  + google_project_iam_member.tpu_sa_log_writer
  + google_storage_bucket.checkpoints
  + google_storage_bucket_iam_member.sa_bucket_access
  + google_tpu_v2_vm.training
```

### Step 4 — Apply

Create all resources:

```bash
terraform apply -var-file=environments/dev/terraform.tfvars
```

Type `yes` when prompted. The TPU VM creation takes 3–5 minutes.

When complete, Terraform prints the outputs:

```
Outputs:

tpu_name              = "tunix-dpo-v5e"
tpu_state             = "READY"
bucket_url            = "gs://your-project-tunix-checkpoints"
service_account_email = "tunix-tpu-sa@your-project.iam.gserviceaccount.com"
ssh_command           = "gcloud compute tpus tpu-vm ssh tunix-dpo-v5e --zone=us-central2-b ..."
scp_command           = "gcloud compute tpus tpu-vm scp --recurse ./ tunix-dpo-v5e:~/tunix-dpo/ ..."
```

### Step 5 — Update configs/dpo_v5e.yaml

Copy the `bucket_url` output into your training config:

```yaml
infra:
  gcs_bucket: "gs://your-project-tunix-checkpoints"  # ← paste bucket_url here
```

### Step 6 — Copy project files to the VM

Use the `scp_command` output directly:

```bash
gcloud compute tpus tpu-vm scp \
  --recurse ./ tunix-dpo-v5e:~/tunix-dpo/ \
  --zone=us-central1-a \
  --project=your-project-id
```

### Step 7 — SSH in and install the package

```bash
gcloud compute tpus tpu-vm ssh tunix-dpo-v5e \
  --zone=us-central1-a \
  --project=your-project-id

# Once on the VM:
cd ~/tunix-dpo
pip install -e ".[training]"
pip install "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

# Verify 8 chips are visible
python -c "import jax; print(len(jax.devices()), 'TPU devices')"
```

### Step 8 — Run training

```bash
# On the TPU VM:
tunix-train --config configs/dpo_v5e.yaml
```

### Step 9 — Destroy after training

**TPU VMs bill by the minute. Destroy immediately when training is done.**

```bash
# From your local machine:
terraform destroy -var-file=environments/dev/terraform.tfvars
```

Type `yes`. This deletes the TPU VM and service account.  
The GCS bucket is **not** deleted (to protect your checkpoints).  
To also delete the bucket: set `force_destroy = true` in main.tf, then re-apply before destroying.

---

## Useful Terraform commands

```bash
# Re-read current state of all resources (without applying changes)
terraform refresh -var-file=environments/dev/terraform.tfvars

# Show the current state
terraform show

# Show a specific output value
terraform output bucket_url
terraform output ssh_command

# Validate syntax without connecting to GCP
terraform validate

# Format all .tf files
terraform fmt

# Destroy only the TPU VM (keep bucket and SA)
terraform destroy \
  -target=google_tpu_v2_vm.training \
  -var-file=environments/dev/terraform.tfvars

# Re-create only the TPU VM (after a targeted destroy)
terraform apply \
  -target=google_tpu_v2_vm.training \
  -var-file=environments/dev/terraform.tfvars
```

---

## Using remote state (team setup)

Uncomment the `backend "gcs"` block in `main.tf` and create the state bucket first:

```bash
gsutil mb -l us-central1 gs://YOUR_PROJECT_ID-terraform-state
```

Then re-initialise:

```bash
terraform init \
  -backend-config="bucket=YOUR_PROJECT_ID-terraform-state" \
  -backend-config="prefix=tunix-dpo/phase2"
```

Remote state lets multiple people (or CI jobs) share the same infrastructure view without conflicts.

---

## Cost notes

| Resource | Approximate cost |
|----------|-----------------|
| TPU v5e-8 on-demand | ~$12–16 / hour |
| TPU v5e-8 preemptible | ~$4–5 / hour |
| GCS bucket | ~$0.02 / GB / month |
| Service account, IAM | Free |

Set `preemptible = true` in `terraform.tfvars` to use Spot pricing.
The training run typically completes in 3–6 hours for one epoch on HH-RLHF.
Always run `terraform destroy` immediately after training.

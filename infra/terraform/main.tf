# ─── Tunix DPO :: GCP Infrastructure ────────────────────────────────────────
# Creates the TPU v5e-8 VM, GCS bucket, service account, and IAM bindings.
#
# Usage:
#   1. cd infra/terraform
#   2. cp environments/dev/terraform.tfvars.example environments/dev/terraform.tfvars
#   3. Edit environments/dev/terraform.tfvars with your project_id
#   4. terraform init
#   5. terraform plan  -var-file=environments/dev/terraform.tfvars
#   6. terraform apply -var-file=environments/dev/terraform.tfvars
#   7. ⚠️  DELETE ASAP after training finishes:
#      terraform destroy -var-file=environments/dev/terraform.tfvars
# ────────────────────────────────────────────────────────────────────────────

terraform {
  required_version = ">= 1.5"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
  zone    = var.zone
}

# ─── Required APIs ──────────────────────────────────────────────────────────
resource "google_project_service" "tpu" {
  service            = "tpu.googleapis.com"
  disable_on_destroy = false
}

resource "google_project_service" "storage" {
  service            = "storage.googleapis.com"
  disable_on_destroy = false
}

resource "google_project_service" "compute" {
  service            = "compute.googleapis.com"
  disable_on_destroy = false
}

resource "google_project_service" "iam" {
  service            = "iam.googleapis.com"
  disable_on_destroy = false
}

# ─── Service account for the TPU VM ─────────────────────────────────────────
resource "google_service_account" "tpu_sa" {
  account_id   = "tunix-tpu-sa"
  display_name = "Tunix DPO TPU service account"
  description  = "Service account used by the TPU VM to read/write checkpoints"

  depends_on = [google_project_service.iam]
}

# Grant access to write checkpoints to GCS
resource "google_project_iam_member" "tpu_sa_storage" {
  project = var.project_id
  role    = "roles/storage.objectAdmin"
  member  = "serviceAccount:${google_service_account.tpu_sa.email}"
}

# Grant access to write logs
resource "google_project_iam_member" "tpu_sa_logging" {
  project = var.project_id
  role    = "roles/logging.logWriter"
  member  = "serviceAccount:${google_service_account.tpu_sa.email}"
}

# ─── GCS bucket for training checkpoints ────────────────────────────────────
# Co-located with the TPU region so writes are free and low-latency.
# force_destroy = false so terraform destroy will NOT delete the bucket if it
# contains objects (i.e. trained checkpoints) — protects expensive results.
resource "google_storage_bucket" "checkpoints" {
  name          = "${var.project_id}-tunix-checkpoints"
  location      = var.region
  storage_class = "STANDARD"

  uniform_bucket_level_access = true
  public_access_prevention    = "enforced"

  versioning {
    enabled = true
  }

  lifecycle_rule {
    action {
      type = "SetStorageClass"
      storage_class = "NEARLINE"
    }
    condition {
      age = 30  # tier down checkpoints older than 30 days
    }
  }

  force_destroy = false  # protect against accidental deletion

  depends_on = [google_project_service.storage]
}

# ─── TPU VM ────────────────────────────────────────────────────────────────
# v5e-8 in us-west4-a — Google's canonical zone for TPU v5e.
# The "v5litepod-8" name is the gcloud / Compute Engine convention; the
# Terraform google_tpu_v2_vm resource accepts the same accelerator types.
resource "google_tpu_v2_vm" "training" {
  name             = var.tpu_name
  zone             = var.zone
  description      = "Tunix DPO training VM (Gemma 3 1B IT, DPO)"
  accelerator_type = var.accelerator_type
  runtime_version  = var.runtime_version

  network_config {
    network    = "default"
    subnetwork = "default"
    enable_external_ips = true
  }

  service_account {
    email = google_service_account.tpu_sa.email
    scope = ["https://www.googleapis.com/auth/cloud-platform"]
  }

  metadata = {
    startup-script = file("${path.module}/scripts/startup.sh")
    bucket         = google_storage_bucket.checkpoints.name
  }

  labels = {
    project = "tunix-dpo"
    env     = var.env
    model   = "gemma-3-1b-it"
  }

  depends_on = [
    google_project_service.tpu,
    google_project_iam_member.tpu_sa_storage,
  ]
}

terraform {
  required_version = ">= 1.6"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }

  # Uncomment to store state in GCS (recommended for team use).
  # The bucket must already exist before running terraform init.
  # backend "gcs" {
  #   bucket = "YOUR_PROJECT_ID-terraform-state"
  #   prefix = "tunix-dpo/phase2"
  # }
}

provider "google" {
  project = var.project_id
  region  = var.region
  zone    = var.zone
}

# ── Local values ──────────────────────────────────────────────────────────────

locals {
  bucket_name = var.bucket_name != "" ? var.bucket_name : "${var.project_id}-tunix-checkpoints"
  sa_email    = "${var.service_account_id}@${var.project_id}.iam.gserviceaccount.com"
}

# ── Required APIs ─────────────────────────────────────────────────────────────
# Enables the four APIs the training pipeline needs.
# Set enable_apis = false if your project already has them enabled.

resource "google_project_service" "tpu" {
  count   = var.enable_apis ? 1 : 0
  project = var.project_id
  service = "tpu.googleapis.com"

  disable_dependent_services = false
  disable_on_destroy         = false
}

resource "google_project_service" "storage" {
  count   = var.enable_apis ? 1 : 0
  project = var.project_id
  service = "storage.googleapis.com"

  disable_dependent_services = false
  disable_on_destroy         = false
}

resource "google_project_service" "compute" {
  count   = var.enable_apis ? 1 : 0
  project = var.project_id
  service = "compute.googleapis.com"

  disable_dependent_services = false
  disable_on_destroy         = false
}

resource "google_project_service" "iam" {
  count   = var.enable_apis ? 1 : 0
  project = var.project_id
  service = "iam.googleapis.com"

  disable_dependent_services = false
  disable_on_destroy         = false
}

# ── Service account ───────────────────────────────────────────────────────────
# Dedicated identity for the TPU VM. Grants only what training needs:
# GCS read/write (checkpoints + logs) and TPU usage.

resource "google_service_account" "tpu_sa" {
  project      = var.project_id
  account_id   = var.service_account_id
  display_name = "Tunix DPO TPU training service account"

  depends_on = [google_project_service.iam]
}

resource "google_project_iam_member" "tpu_sa_storage" {
  project = var.project_id
  role    = "roles/storage.objectAdmin"
  member  = "serviceAccount:${google_service_account.tpu_sa.email}"
}

resource "google_project_iam_member" "tpu_sa_tpu_viewer" {
  project = var.project_id
  role    = "roles/tpu.viewer"
  member  = "serviceAccount:${google_service_account.tpu_sa.email}"
}

resource "google_project_iam_member" "tpu_sa_log_writer" {
  project = var.project_id
  role    = "roles/logging.logWriter"
  member  = "serviceAccount:${google_service_account.tpu_sa.email}"
}

# ── GCS bucket ────────────────────────────────────────────────────────────────
# Single-region bucket co-located with the TPU VM.
# Co-location is critical: cross-region checkpoint writes are billed
# and add significant latency to each checkpoint step.

resource "google_storage_bucket" "checkpoints" {
  project  = var.project_id
  name     = local.bucket_name
  location = var.region

  # Delete bucket contents before destroying (safe for ephemeral training runs;
  # set to false if you want Terraform to refuse destruction when checkpoints exist).
  force_destroy = false

  uniform_bucket_level_access = true
  public_access_prevention    = "enforced"

  versioning {
    enabled = false
  }

  lifecycle_rule {
    condition {
      age = 90
    }
    action {
      type = "Delete"
    }
  }

  labels = var.labels

  depends_on = [google_project_service.storage]
}

resource "google_storage_bucket_iam_member" "sa_bucket_access" {
  bucket = google_storage_bucket.checkpoints.name
  role   = "roles/storage.objectAdmin"
  member = "serviceAccount:${google_service_account.tpu_sa.email}"
}

# ── TPU VM ────────────────────────────────────────────────────────────────────
# The google_tpu_v2_vm resource (provider >= 5.0) maps directly to the
# gcloud compute tpus tpu-vm create command used in tpu_provision.sh.

resource "google_tpu_v2_vm" "training" {
  project  = var.project_id
  name     = var.tpu_name
  zone     = var.zone

  runtime_version  = var.runtime_version
  accelerator_type = var.accelerator_type

  # Attach the dedicated service account so the VM can write to GCS
  # without user credentials present.
  service_account {
    email  = google_service_account.tpu_sa.email
    scope  = ["https://www.googleapis.com/auth/cloud-platform"]
  }

  # Boot disk
  boot_disk {
    # Zonal persistent disk in the same zone as the TPU VM.
    disk_size_gb  = var.boot_disk_size_gb
    disk_type = "pd-ssd"
  }

  # Preemptible (Spot) VMs are ~70% cheaper but can be interrupted.
  scheduling_config {
    preemptible = var.preemptible
  }

  labels = var.labels

  # Ensure the API is enabled and the service account exists before creating.
  depends_on = [
    google_project_service.tpu,
    google_project_service.compute,
    google_service_account.tpu_sa,
    google_storage_bucket.checkpoints,
  ]
}

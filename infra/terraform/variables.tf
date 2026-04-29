variable "project_id" {
  description = "GCP project ID."
  type        = string
}

variable "region" {
  description = "GCP region for GCS bucket and networking. Must match the zone's region."
  type        = string
  default     = "us-west4"
}

variable "zone" {
  description = "GCP zone where the TPU VM is created. Must support v5e hardware."
  type        = string
  default     = "us-west4-a"

  validation {
    condition = contains([
      "us-central1-a",
      "us-south1-a",
      "us-west1-c",
      "us-west4-a",
      "europe-west4-b",
    ], var.zone)
    error_message = "Zone must be one of the zones where TPU v5e hardware is available."
  }
}

variable "tpu_name" {
  description = "Name for the TPU VM resource."
  type        = string
  default     = "tunix-dpo-v5e"
}

variable "accelerator_type" {
  description = "TPU accelerator type. v5e-8 = one v5e pod slice with 8 chips."
  type        = string
  default     = "v5e-8"
}

variable "runtime_version" {
  description = "TPU software runtime version."
  type        = string
  default     = "tpu-vm-tf-2.16.1-pjrt"
}

variable "boot_disk_size_gb" {
  description = "Boot disk size in GB for the TPU VM."
  type        = number
  default     = 200
}

variable "preemptible" {
  description = "Whether to use a preemptible (cheaper but interruptible) TPU VM."
  type        = bool
  default     = false
}

variable "bucket_name" {
  description = "GCS bucket name for checkpoints and logs. Defaults to <project_id>-tunix-checkpoints."
  type        = string
  default     = ""
}

variable "service_account_id" {
  description = "ID for the training service account (the local part before @)."
  type        = string
  default     = "tunix-tpu-sa"
}

variable "enable_apis" {
  description = "Whether Terraform should enable the required GCP APIs. Set false if APIs are already enabled."
  type        = bool
  default     = true
}

variable "labels" {
  description = "Labels to apply to all created resources."
  type        = map(string)
  default = {
    project     = "tunix-dpo"
    managed_by  = "terraform"
  }
}

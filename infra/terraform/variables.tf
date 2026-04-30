# ─── Tunix DPO :: Terraform variables ───────────────────────────────────────

variable "project_id" {
  description = "GCP project ID."
  type        = string
}

variable "region" {
  description = "GCP region. TPU quota is regional (lives at this level)."
  type        = string
  default     = "us-west4"
}

variable "zone" {
  description = <<EOT
GCP zone where the TPU VM is created. The project default is us-west4-a — the
canonical zone Google uses in its TPU v5e training documentation. Other
documented v5e zones include europe-west4-a and asia-southeast1-c if you need
them, but the project's tooling and bucket region default to us-west4.
EOT
  type    = string
  default = "us-west4-a"

  validation {
    condition = contains(
      [
        "us-west4-a",        # primary (project default)
        "europe-west4-a",    # EU alternative
        "asia-southeast1-c", # APAC alternative
      ],
      var.zone,
    )
    error_message = "var.zone must be a documented TPU v5e zone (us-west4-a, europe-west4-a, or asia-southeast1-c)."
  }
}

variable "tpu_name" {
  description = "Name of the TPU VM."
  type        = string
  default     = "tunix-dpo-v5e"
}

variable "accelerator_type" {
  description = "TPU accelerator type. v5litepod-4 = v5e with 8 chips."
  type        = string
  default     = "v5litepod-4"
}

variable "runtime_version" {
  description = "TPU runtime version. v2-alpha-tpuv5-lite is the v5e runtime."
  type        = string
  default     = "v2-alpha-tpuv5-lite"
}

variable "env" {
  description = "Environment label (dev | staging | prod)."
  type        = string
  default     = "dev"
}

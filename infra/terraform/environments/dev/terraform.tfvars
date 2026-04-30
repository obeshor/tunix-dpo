# Copy this file to terraform.tfvars and fill in your project_id.
# terraform.tfvars is gitignored — never commit it.

project_id       = "tpusprint"
region           = "us-west4"
zone             = "us-west4-a"
tpu_name         = "tunix-dpo-v5e-dev"
accelerator_type = "v5litepod-4"
runtime_version  = "v2-alpha-tpuv5-lite"
env              = "dev"

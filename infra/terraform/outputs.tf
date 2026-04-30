# ─── Tunix DPO :: Terraform outputs ─────────────────────────────────────────

output "tpu_name" {
  description = "Name of the provisioned TPU VM."
  value       = google_tpu_v2_vm.training.name
}

output "tpu_zone" {
  description = "Zone where the TPU VM lives."
  value       = google_tpu_v2_vm.training.zone
}

output "service_account_email" {
  description = "Email of the TPU service account."
  value       = google_service_account.tpu_sa.email
}

output "bucket_name" {
  description = "Name of the GCS checkpoint bucket."
  value       = google_storage_bucket.checkpoints.name
}

output "bucket_url" {
  description = "gs:// URL of the checkpoint bucket."
  value       = "gs://${google_storage_bucket.checkpoints.name}"
}

output "ssh_command" {
  description = "Command to SSH into the TPU VM."
  value = format(
    "gcloud compute tpus tpu-vm ssh %s --project=%s --zone=%s",
    google_tpu_v2_vm.training.name,
    var.project_id,
    var.zone,
  )
}

output "scp_command" {
  description = "Example SCP command to copy code onto the TPU VM."
  value = format(
    "gcloud compute tpus tpu-vm scp --recurse ./tunix_dpo %s:~/ --project=%s --zone=%s",
    google_tpu_v2_vm.training.name,
    var.project_id,
    var.zone,
  )
}

output "destroy_warning" {
  description = "Reminder to tear down the TPU VM after training."
  value       = "⚠️  Run 'terraform destroy' immediately after training finishes — TPU v5e-8 costs $12–16/hr."
}

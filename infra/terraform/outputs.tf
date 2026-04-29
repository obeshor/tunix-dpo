output "tpu_name" {
  description = "Name of the provisioned TPU VM."
  value       = google_tpu_v2_vm.training.name
}

output "tpu_zone" {
  description = "Zone where the TPU VM was created."
  value       = var.zone
}

output "tpu_state" {
  description = "Current state of the TPU VM (READY when provisioning succeeds)."
  value       = google_tpu_v2_vm.training.state
}

output "tpu_network_endpoints" {
  description = "Internal IP addresses of the TPU VM chips."
  value       = google_tpu_v2_vm.training.network_endpoints
}

output "bucket_name" {
  description = "GCS bucket name for checkpoints and logs."
  value       = google_storage_bucket.checkpoints.name
}

output "bucket_url" {
  description = "Full GCS URL for use in configs/dpo_v5e.yaml."
  value       = "gs://${google_storage_bucket.checkpoints.name}"
}

output "service_account_email" {
  description = "Email of the TPU training service account."
  value       = google_service_account.tpu_sa.email
}

output "ssh_command" {
  description = "Command to SSH into the TPU VM."
  value       = "gcloud compute tpus tpu-vm ssh ${google_tpu_v2_vm.training.name} --zone=${var.zone} --project=${var.project_id}"
}

output "scp_command" {
  description = "Command to copy the project files to the TPU VM."
  value       = "gcloud compute tpus tpu-vm scp --recurse ./ ${google_tpu_v2_vm.training.name}:~/tunix-dpo/ --zone=${var.zone} --project=${var.project_id}"
}

output "destroy_command" {
  description = "Reminder: run this after training to stop billing."
  value       = "terraform destroy -var-file=environments/dev/terraform.tfvars"
}

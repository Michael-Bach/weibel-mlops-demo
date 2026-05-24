variable "kubeconfig_path" {
  description = "Absolute path to kubeconfig file"
  type        = string
  default     = "~/.kube/config"
}

variable "kube_context" {
  description = "Kubernetes context name from kubeconfig"
  type        = string
}

variable "grafana_admin_password" {
  description = "Grafana admin password — provide via tfvars or Vault, never hardcode"
  type        = string
  sensitive   = true
}

variable "mlflow_tracking_uri" {
  description = "MLflow tracking server URI"
  type        = string
  default     = "http://mlflow-server:5000"
}

variable "nfs_server" {
  description = "NFS server hostname or IP for model artifact storage"
  type        = string
}

variable "nfs_model_path" {
  description = "NFS export path for model artifacts"
  type        = string
  default     = "/exports/model-store"
}

variable "model_store_size_gi" {
  description = "Model artifact PersistentVolume size in GiB"
  type        = number
  default     = 50
}

variable "prometheus_retention_days" {
  description = "Prometheus metric retention in days"
  type        = number
  default     = 30
}

variable "storage_class" {
  description = "Kubernetes StorageClass name for PVCs (e.g. 'standard', 'gp2', 'nfs-client')"
  type        = string
  default     = "standard"
}

# Terraform — MLOps infrastructure provisioning.
#
# Provisions:
#   - Kubernetes namespaces (ml-training, ml-serving, monitoring)
#   - NVIDIA GPU device plugin (DaemonSet via Helm)
#   - kube-prometheus-stack (Prometheus + Grafana + Alertmanager)
#   - MLflow tracking server
#   - Persistent volume for model artifacts
#
# Provider: generic Kubernetes + Helm. Adapt the provider config for your
# cluster — kubeconfig file for on-prem, or use the cloud provider's Terraform
# module (google_container_cluster, aws_eks_cluster, etc.).
#
# Usage:
#   terraform init
#   terraform workspace new staging
#   terraform plan  -var-file=env/staging.tfvars
#   terraform apply -var-file=env/staging.tfvars

terraform {
  required_version = ">= 1.6"
  required_providers {
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.30"
    }
    helm = {
      source  = "hashicorp/helm"
      version = "~> 2.13"
    }
  }
  # State backend — use S3 or GCS for team environments.
  # Uncomment and configure for production:
  # backend "s3" {
  #   bucket = "your-terraform-state"
  #   key    = "weibel-mlops/terraform.tfstate"
  #   region = "eu-west-1"
  # }
}

provider "kubernetes" {
  config_path    = var.kubeconfig_path
  config_context = var.kube_context
}

provider "helm" {
  kubernetes {
    config_path    = var.kubeconfig_path
    config_context = var.kube_context
  }
}

# ── Namespaces ────────────────────────────────────────────────────────────────

locals {
  namespaces = {
    training   = "ml-training"
    serving    = "ml-serving"
    monitoring = "monitoring"
  }
}

resource "kubernetes_namespace" "mlops" {
  for_each = local.namespaces
  metadata {
    name = each.value
    labels = {
      purpose  = each.key
      managed  = "terraform"
      project  = "weibel-radar"
    }
  }
}

# ── NVIDIA GPU device plugin ──────────────────────────────────────────────────
# Exposes nvidia.com/gpu as a schedulable resource. Required for GPU Jobs.
# Runs as a DaemonSet on nodes with the GPU node label.

resource "helm_release" "nvidia_device_plugin" {
  name       = "nvidia-device-plugin"
  repository = "https://nvidia.github.io/k8s-device-plugin"
  chart      = "nvidia-device-plugin"
  namespace  = "kube-system"
  version    = "0.16.1"

  set {
    name  = "compatWithCPUManager"
    value = "true"
  }
  set {
    name  = "nodeSelector.node-type"
    value = "gpu-training"
  }
}

# ── kube-prometheus-stack (Prometheus + Grafana + Alertmanager) ───────────────

resource "helm_release" "prometheus_stack" {
  name       = "kube-prometheus-stack"
  repository = "https://prometheus-community.github.io/helm-charts"
  chart      = "kube-prometheus-stack"
  namespace  = kubernetes_namespace.mlops["monitoring"].metadata[0].name
  version    = "60.3.0"
  timeout    = 600

  values = [yamlencode({
    grafana = {
      adminPassword = var.grafana_admin_password
      persistence = {
        enabled          = true
        size             = "10Gi"
        storageClassName = var.storage_class
      }
    }
    prometheus = {
      prometheusSpec = {
        retention = "${var.prometheus_retention_days}d"
        storageSpec = {
          volumeClaimTemplate = {
            spec = {
              storageClassName = var.storage_class
              resources        = { requests = { storage = "50Gi" } }
            }
          }
        }
        additionalScrapeConfigs = []
      }
    }
    alertmanager = {
      enabled = true
    }
  })]

  depends_on = [kubernetes_namespace.mlops]
}

# ── MLflow tracking server ────────────────────────────────────────────────────

resource "helm_release" "mlflow" {
  name      = "mlflow"
  chart     = "${path.module}/../../k8s/helm"
  namespace = kubernetes_namespace.mlops["training"].metadata[0].name

  set {
    name  = "mlflow.trackingUri"
    value = var.mlflow_tracking_uri
  }

  depends_on = [kubernetes_namespace.mlops]
}

# ── Persistent volume for model artifacts ─────────────────────────────────────

resource "kubernetes_persistent_volume" "model_store" {
  metadata {
    name   = "model-store"
    labels = { managed = "terraform", project = "weibel-radar" }
  }
  spec {
    capacity                         = { storage = "${var.model_store_size_gi}Gi" }
    access_modes                     = ["ReadWriteMany"]
    persistent_volume_reclaim_policy = "Retain"
    nfs {
      server = var.nfs_server
      path   = var.nfs_model_path
    }
  }
}

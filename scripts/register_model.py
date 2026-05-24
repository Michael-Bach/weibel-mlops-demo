"""
MLflow model registry — promotion flow.

Stages: run artifact → Staging (Candidate) → Production

Promotion is gated on val_accuracy thresholds. The script is meant to be
called by a human operator or an automated validation step, never by the
training loop itself — that separation is intentional.

Usage:
    # Register a training run as a Staging candidate:
    python scripts/register_model.py --run-id <mlflow-run-id>

    # Promote an already-staged version to Production:
    python scripts/register_model.py --run-id <mlflow-run-id> --promote-to production

    # List current registry state:
    python scripts/register_model.py --list
"""

import argparse
import sys

import mlflow
from mlflow import MlflowClient

MODEL_NAME = "radar-classifier"

# Accuracy gates — adjust for production requirements.
# Defense note: conservative thresholds are intentional. A radar classifier
# that degrades gracefully under corner-case signals is preferred over one
# that hits 99% on clean synthetic data but has no validated floor.
THRESHOLD_STAGING = 0.90
THRESHOLD_PRODUCTION = 0.95


def get_run_metrics(run_id: str, client: MlflowClient) -> dict:
    return client.get_run(run_id).data.metrics


def register_candidate(run_id: str, client: MlflowClient) -> str:
    """Register run artifact in the MLflow registry at Staging stage."""
    model_uri = f"runs:/{run_id}/model_best.pt"
    mv = mlflow.register_model(model_uri, MODEL_NAME)
    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=mv.version,
        stage="Staging",
        archive_existing_versions=False,
    )
    print(f"Registered {MODEL_NAME} v{mv.version} → Staging (run_id={run_id})")
    return mv.version


def promote_to_production(version: str, metrics: dict, client: MlflowClient) -> None:
    acc = metrics.get("best_val_accuracy", 0.0)
    if acc < THRESHOLD_PRODUCTION:
        print(
            f"BLOCKED: best_val_accuracy={acc:.4f} < threshold={THRESHOLD_PRODUCTION}\n"
            f"Run hardware-in-the-loop validation and re-evaluate before promoting."
        )
        sys.exit(1)
    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=version,
        stage="Production",
        archive_existing_versions=True,  # auto-archive the previous Production version
    )
    print(f"Promoted {MODEL_NAME} v{version} → Production (val_accuracy={acc:.4f})")


def list_registry(client: MlflowClient) -> None:
    try:
        versions = client.get_latest_versions(MODEL_NAME)
    except Exception:
        print(f"No model '{MODEL_NAME}' registered yet.")
        return
    if not versions:
        print("Registry is empty.")
        return
    print(f"\n{'Version':<10} {'Stage':<15} {'Run ID':<35} {'Created'}")
    print("-" * 80)
    for v in versions:
        created = v.creation_timestamp
        print(f"{v.version:<10} {v.current_stage:<15} {v.run_id:<35} {created}")


def main() -> None:
    parser = argparse.ArgumentParser(description="MLflow model registry promotion tool")
    parser.add_argument("--run-id", help="MLflow run ID to register")
    parser.add_argument(
        "--promote-to",
        choices=["staging", "production"],
        default="staging",
        help="Target stage (default: staging)",
    )
    parser.add_argument("--list", action="store_true", help="List current registry state")
    args = parser.parse_args()

    client = MlflowClient()

    if args.list:
        list_registry(client)
        return

    if not args.run_id:
        parser.error("--run-id is required unless --list is specified")

    metrics = get_run_metrics(args.run_id, client)
    print(f"Run metrics: {metrics}")

    acc = metrics.get("best_val_accuracy", 0.0)
    if acc < THRESHOLD_STAGING:
        print(f"BLOCKED: best_val_accuracy={acc:.4f} < staging threshold={THRESHOLD_STAGING}")
        sys.exit(1)

    version = register_candidate(args.run_id, client)

    if args.promote_to == "production":
        promote_to_production(version, metrics, client)


if __name__ == "__main__":
    main()

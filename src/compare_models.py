#!/usr/bin/env python3
import os
import sys
from typing import Dict, List

try:
    import mlflow
    from mlflow.tracking import MlflowClient
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    print("ERROR: MLflow not available. Please install mlflow.")
    sys.exit(1)


def get_experiment_runs(experiment_name: str = "mlsecops-fraud-detection") -> List[Dict]:
    """Get all runs from the experiment, sorted by start time"""
    client = MlflowClient()

    experiment = client.get_experiment_by_name(experiment_name)
    if not experiment:
        print(f"ERROR: Experiment '{experiment_name}' not found")
        sys.exit(1)

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
        max_results=10
    )

    return runs


def get_latest_model_runs(experiment_name: str = "mlsecops-fraud-detection") -> tuple:
    """Get the 2 latest runs from current experiment."""
    runs = get_experiment_runs(experiment_name)

    rf_run = None
    xgb_run = None

    for run in runs:
        model_type = run.data.tags.get("model_type", "")

        if "Random Forest" in model_type and not rf_run:
            rf_run = run
        elif "XGBoost" in model_type and not xgb_run:
            xgb_run = run

        if rf_run and xgb_run:
            break

    return rf_run, xgb_run


def compare_models(rf_run, xgb_run) -> Dict:
    """Compare models based on fixed_recall metric."""

    # Extract metrics
    rf_metrics = rf_run.data.metrics
    xgb_metrics = xgb_run.data.metrics

    rf_recall = rf_metrics.get("fixed_recall", 0.0)
    xgb_recall = xgb_metrics.get("fixed_recall", 0.0)

    rf_roc_auc = rf_metrics.get("roc_auc", 0.0)
    xgb_roc_auc = xgb_metrics.get("roc_auc", 0.0)

    print("=" * 70)
    print("MODEL COMPARISON RESULTS")
    print("=" * 70)
    print(f"\nMetric Priority: Fixed Recall @ FPR=5%")
    print(f"\n{'Model':<20} {'Fixed Recall':<15} {'ROC-AUC':<12} {'Run ID'}")
    print("-" * 70)
    print(f"{'Random Forest':<20} {rf_recall:<15.4f} {rf_roc_auc:<12.4f} {rf_run.info.run_id[:8]}")
    print(f"{'XGBoost':<20} {xgb_recall:<15.4f} {xgb_roc_auc:<12.4f} {xgb_run.info.run_id[:8]}")
    print("-" * 70)

    # Compare based on fixed_recall
    if rf_recall > xgb_recall:
        winner_run = rf_run
        winner_name = "Random Forest"
        winner_recall = rf_recall
        diff = rf_recall - xgb_recall
    else:
        winner_run = xgb_run
        winner_name = "XGBoost"
        winner_recall = xgb_recall
        diff = xgb_recall - rf_recall

    print(f"\n WINNER: {winner_name}")
    print(f"   Fixed Recall: {winner_recall:.4f}")
    print(f"   Better by: {diff:.4f} ({diff*100:.2f}%)")
    print("=" * 70)

    return {
        "winner_run": winner_run,
        "winner_name": winner_name,
        "winner_recall": winner_recall,
        "rf_recall": rf_recall,
        "xgb_recall": xgb_recall
    }


def register_model(run, model_name: str, registry_name: str = "fraud-detection-model"):
    """Register the winning model to MLflow Model Registry."""
    client = MlflowClient()

    # Model artifact path in MLflow run
    model_uri = f"runs:/{run.info.run_id}/random_forest_model" if "Random Forest" in model_name else f"runs:/{run.info.run_id}/xgboost_model"

    print(f"\n Registering model to Model Registry...")
    print(f"   Model: {model_name}")
    print(f"   Run ID: {run.info.run_id}")
    print(f"   Registry Name: {registry_name}")

    try:
        # Register model
        model_version = mlflow.register_model(model_uri, registry_name)

        # Add description
        client.update_model_version(
            name=registry_name,
            version=model_version.version,
            description=f"{model_name} - Auto-selected based on Fixed Recall @ FPR=5%"
        )

        # Add tags
        client.set_model_version_tag(
            name=registry_name,
            version=model_version.version,
            key="model_type",
            value=model_name
        )

        client.set_model_version_tag(
            name=registry_name,
            version=model_version.version,
            key="selection_metric",
            value="fixed_recall"
        )

        print(f"   Model registered successfully!")
        print(f"   Version: {model_version.version}")
        print(f"   Stage: {model_version.current_stage}")

        return model_version

    except Exception as e:
        print(f"ERROR: Failed to register model - {e}")
        return None


def main():
    print("Starting model comparison workflow...")

    # Setup MLflow
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        print("ERROR: MLFLOW_TRACKING_URI environment variable not set")
        sys.exit(1)

    mlflow.set_tracking_uri(tracking_uri)
    print(f"MLflow Tracking URI: {tracking_uri}\n")

    # Get latest runs
    print("Fetching latest model runs from MLflow...")
    rf_run, xgb_run = get_latest_model_runs()

    if not rf_run or not xgb_run:
        print("ERROR: Could not find both RF and XGBoost runs")
        print(f"  Random Forest run: {'Found' if rf_run else 'Not found'}")
        print(f"  XGBoost run: {'Found' if xgb_run else 'Not found'}")
        sys.exit(1)

    # Compare models
    result = compare_models(rf_run, xgb_run)

    # Register winner
    model_version = register_model(
        result["winner_run"],
        result["winner_name"]
    )

    if model_version:
        print("\nModel comparison and registration completed successfully!")
        return 0
    else:
        print("\nModel registration failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())

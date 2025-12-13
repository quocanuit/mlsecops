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
    """Compare models using normalized metrics and aggregate scoring."""

    # Extract metrics
    rf_metrics = rf_run.data.metrics
    xgb_metrics = xgb_run.data.metrics

    # Get all metrics (with fallback to 0.0)
    rf_recall = rf_metrics.get("recall", 0.0)
    xgb_recall = xgb_metrics.get("recall", 0.0)
    
    rf_roc_auc = rf_metrics.get("roc_auc", 0.0)
    xgb_roc_auc = xgb_metrics.get("roc_auc", 0.0)
    
    rf_log_loss = rf_metrics.get("log_loss", 1.0)
    xgb_log_loss = xgb_metrics.get("log_loss", 1.0)
    
    rf_f1 = rf_metrics.get("f1_score", 0.0)
    xgb_f1 = xgb_metrics.get("f1_score", 0.0)
    
    rf_precision = rf_metrics.get("precision", 0.0)
    xgb_precision = xgb_metrics.get("precision", 0.0)
    
    rf_time = rf_metrics.get("total_time_seconds", 0.0)
    xgb_time = xgb_metrics.get("total_time_seconds", 0.0)

    # Store in lists for normalization
    recall_list = [rf_recall, xgb_recall]
    roc_auc_list = [rf_roc_auc, xgb_roc_auc]
    log_loss_list = [rf_log_loss, xgb_log_loss]
    f1_list = [rf_f1, xgb_f1]
    time_list = [rf_time, xgb_time]

    # Normalize metrics (0-1 scale, higher is better)
    def normalize(value, min_val, max_val):
        if max_val == min_val:
            return 0.5  # Equal performance
        return (value - min_val) / (max_val - min_val)
    
    def normalize_inverse(value, min_val, max_val):
        """For metrics where lower is better (log_loss, time)"""
        if max_val == min_val:
            return 0.5
        return (max_val - value) / (max_val - min_val)

    # Normalize RF metrics
    rf_norm_recall = normalize(rf_recall, min(recall_list), max(recall_list))
    rf_norm_roc_auc = normalize(rf_roc_auc, min(roc_auc_list), max(roc_auc_list))
    rf_norm_log_loss = normalize_inverse(rf_log_loss, min(log_loss_list), max(log_loss_list))
    rf_norm_f1 = normalize(rf_f1, min(f1_list), max(f1_list))
    rf_norm_time = normalize_inverse(rf_time, min(time_list), max(time_list))

    # Normalize XGBoost metrics
    xgb_norm_recall = normalize(xgb_recall, min(recall_list), max(recall_list))
    xgb_norm_roc_auc = normalize(xgb_roc_auc, min(roc_auc_list), max(roc_auc_list))
    xgb_norm_log_loss = normalize_inverse(xgb_log_loss, min(log_loss_list), max(log_loss_list))
    xgb_norm_f1 = normalize(xgb_f1, min(f1_list), max(f1_list))
    xgb_norm_time = normalize_inverse(xgb_time, min(time_list), max(time_list))

    # Calculate aggregate scores (weighted average)
    # Weights: Recall=0.30, ROC-AUC=0.25, F1=0.20, Log Loss=0.15, Time=0.10
    rf_aggregate = (0.30 * rf_norm_recall + 0.25 * rf_norm_roc_auc + 
                    0.20 * rf_norm_f1 + 0.15 * rf_norm_log_loss + 0.10 * rf_norm_time)
    
    xgb_aggregate = (0.30 * xgb_norm_recall + 0.25 * xgb_norm_roc_auc + 
                     0.20 * xgb_norm_f1 + 0.15 * xgb_norm_log_loss + 0.10 * xgb_norm_time)

    # Print detailed comparison
    print("=" * 90)
    print("MODEL COMPARISON RESULTS - COMPREHENSIVE ANALYSIS")
    print("=" * 90)
    
    print(f"\n{'Metric':<25} {'Random Forest':<20} {'XGBoost':<20} {'Winner'}")
    print("-" * 90)
    print(f"{'Recall':<25} {rf_recall:<20.4f} {xgb_recall:<20.4f} {'RF' if rf_recall > xgb_recall else 'XGB'}")
    print(f"{'ROC-AUC':<25} {rf_roc_auc:<20.4f} {xgb_roc_auc:<20.4f} {'RF' if rf_roc_auc > xgb_roc_auc else 'XGB'}")
    print(f"{'F1-Score':<25} {rf_f1:<20.4f} {xgb_f1:<20.4f} {'RF' if rf_f1 > xgb_f1 else 'XGB'}")
    print(f"{'Precision':<25} {rf_precision:<20.4f} {xgb_precision:<20.4f} {'RF' if rf_precision > xgb_precision else 'XGB'}")
    print(f"{'Log Loss':<25} {rf_log_loss:<20.4f} {xgb_log_loss:<20.4f} {'RF' if rf_log_loss < xgb_log_loss else 'XGB'}")
    print(f"{'Total Time (s)':<25} {rf_time:<20.4f} {xgb_time:<20.4f} {'RF' if rf_time < xgb_time else 'XGB'}")
    
    print("\n" + "=" * 90)
    print("NORMALIZED SCORES (0-1 scale, higher is better)")
    print("=" * 90)
    print(f"{'Metric':<25} {'RF Normalized':<20} {'XGB Normalized':<20} {'Weight'}")
    print("-" * 90)
    print(f"{'Recall':<25} {rf_norm_recall:<20.4f} {xgb_norm_recall:<20.4f} {'30%'}")
    print(f"{'ROC-AUC':<25} {rf_norm_roc_auc:<20.4f} {xgb_norm_roc_auc:<20.4f} {'25%'}")
    print(f"{'F1-Score':<25} {rf_norm_f1:<20.4f} {xgb_norm_f1:<20.4f} {'20%'}")
    print(f"{'Log Loss (inverted)':<25} {rf_norm_log_loss:<20.4f} {xgb_norm_log_loss:<20.4f} {'15%'}")
    print(f"{'Time (inverted)':<25} {rf_norm_time:<20.4f} {xgb_norm_time:<20.4f} {'10%'}")
    print("-" * 90)
    print(f"{'AGGREGATE SCORE':<25} {rf_aggregate:<20.4f} {xgb_aggregate:<20.4f}")
    
    # Determine winner based on aggregate score
    if rf_aggregate > xgb_aggregate:
        winner_run = rf_run
        winner_name = "Random Forest"
        winner_score = rf_aggregate
        diff = rf_aggregate - xgb_aggregate
    else:
        winner_run = xgb_run
        winner_name = "XGBoost"
        winner_score = xgb_aggregate
        diff = xgb_aggregate - rf_aggregate

    print("\n" + "=" * 90)
    print(f"WINNER: {winner_name}")
    print(f"   Aggregate Score: {winner_score:.4f}")
    print(f"   Better by: {diff:.4f} ({diff*100:.2f}%)")
    print(f"   Run ID: {winner_run.info.run_id}")
    print("=" * 90)

    return {
        "winner_run": winner_run,
        "winner_name": winner_name,
        "winner_score": winner_score,
        "rf_aggregate": rf_aggregate,
        "xgb_aggregate": xgb_aggregate,
        "rf_recall": rf_recall,
        "xgb_recall": xgb_recall,
        "rf_roc_auc": rf_roc_auc,
        "xgb_roc_auc": xgb_roc_auc
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

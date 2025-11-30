import sys
import time
from pathlib import Path
import os

import xgboost as xgb
from mlflow.models import infer_signature
import mlflow

from utils.train_common import (
    load_preprocessed_data,
    create_metadata,
    evaluate_model,
    save_model,
    save_training_report,
    ARTIFACTS_DIR
)
from utils.mlflow_common import (
    setup_mlflow,
    start_run,
    set_tag,
    log_evaluation_metrics,
    log_model_and_params,
    log_training_report,
    print_run_info
)

ROOT = Path(os.getcwd())
RANDOM_STATE_VALUE = 42
FIXED_FPR = 0.05


def load_production_model(model_name: str = "xgboost_model", stage: str = "Production"):
    print(f"Loading production model: {model_name} (stage: {stage})")

    try:
        model_uri = f"models:/{model_name}/{stage}"

        # Load the xgboost model
        model = mlflow.xgboost.load_model(model_uri)

        # Get model version info
        client = mlflow.tracking.MlflowClient()
        model_versions = client.get_latest_versions(model_name, stages=[stage])

        if model_versions:
            version = model_versions[0].version
            run_id = model_versions[0].run_id
            print(f"Loaded model version: {version}, Run ID: {run_id}")
        else:
            print(f"Warning: Could not get model version info")

        return model

    except Exception as e:
        print(f"Error loading production model: {e}")
        print("Falling back to training from scratch...")
        return None


def retrain_model(base_model, X_train, y_train, X_test):
    if base_model is not None:
        print("Performing incremental learning on existing model...")
        # XGBoost supports incremental learning via xgb_model parameter
        model = xgb.XGBClassifier(
            random_state=RANDOM_STATE_VALUE,
            eval_metric='logloss'
        )

        print("Training with incremental learning (continuing from base model)...")
        start_time = time.time()
        model.fit(X_train, y_train, xgb_model=base_model.get_booster())
        train_time = time.time() - start_time
    else:
        print("Training new XGBoost model from scratch...")
        model = xgb.XGBClassifier(random_state=RANDOM_STATE_VALUE, eval_metric='logloss')

        start_time = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start_time

    # Make predictions
    start_time = time.time()
    y_test_pred = model.predict(X_test)
    prediction_time = time.time() - start_time

    y_test_prob = model.predict_proba(X_test)[:, 1]

    # Collect metadata using common function
    metadata = create_metadata(X_train, X_test, model, RANDOM_STATE_VALUE, FIXED_FPR)
    metadata["is_retrained"] = base_model is not None
    metadata["training_type"] = "incremental" if base_model is not None else "from_scratch"

    return model, y_test_pred, y_test_prob, train_time, prediction_time, metadata


def main():
    """Main execution function."""
    print("XGBOOST MODEL RETRAINING")

    setup_mlflow()

    # Start MLflow run
    with start_run(run_name="xgboost_retraining"):
        set_tag("model_type", "XGBoost")
        set_tag("algorithm", "xgboost.XGBClassifier")
        set_tag("training_mode", "retrain")

        # Load production model
        production_model = load_production_model("xgboost_model", "Production")

        # Load preprocessed data (from new production data)
        X_train_resampled, y_train_resampled, X_test_transformed, y_test = load_preprocessed_data()

        # Retrain the model
        model, y_test_pred, y_test_prob, train_time, prediction_time, metadata = retrain_model(
            production_model, X_train_resampled, y_train_resampled, X_test_transformed
        )

        # Evaluate the model
        results_default, results_fixed, fixed_threshold, report_default, report_fixed = evaluate_model(
            "XGBoost (Retrained)", y_test, y_test_pred, y_test_prob, train_time, prediction_time, FIXED_FPR
        )

        # Extract metrics for MLflow logging
        metrics_default = results_default.to_dict(orient='records')[0]
        metrics_fixed = results_fixed.to_dict(orient='records')[0]

        log_evaluation_metrics(
            roc_auc=metrics_default["ROC-AUC Score"],
            logloss=metrics_default["Log Loss"],
            train_time=metrics_default["Training Time (s)"],
            prediction_time=metrics_default["Prediction Time (s)"],
            total_time=metrics_default["Total Time (s)"],
            accuracy_default=metrics_default["Accuracy"],
            prec_default=metrics_default["Precision"],
            rec_default=metrics_default["Recall"],
            f1_default=metrics_default["F1 Score"],
            fpr_default=metrics_default["FPR@threshold"],
            tpr_default=metrics_default["TPR@threshold"],
            fixed_threshold=metrics_fixed["Decision Threshold"],
            accuracy_fixed=metrics_fixed["Accuracy"],
            prec_fixed=metrics_fixed["Precision@threshold"],
            rec_fixed=metrics_fixed["Recall@threshold"],
            f1_fixed=metrics_fixed["F1@threshold"],
            fpr_at=metrics_fixed["FPR@threshold"],
            tpr_at=metrics_fixed["TPR@threshold"]
        )

        # Save model bundle
        model_path = save_model(model, "XGBoost Classifier (Retrained)", 
                                fixed_threshold, FIXED_FPR, metadata, "xgb_model_bundle_retrained.pkl")

        try:
            sig = infer_signature(
                X_train_resampled[:100],
                model.predict_proba(X_train_resampled[:100])[:, 1]
            )
        except Exception:
            sig = None
        input_example = X_train_resampled[:5]

        # Log model to MLflow
        log_model_and_params(
            model, "XGBoost Classifier (Retrained)", metadata, model_path,
            mlflow_model_name="xgboost_model",
            signature=sig, input_example=input_example
        )

        # Save training report
        save_training_report(results_default, results_fixed, report_default, report_fixed, 
                            metadata, "training_report_xgb_retrained.json")

        # Log report to MLflow
        log_training_report(ARTIFACTS_DIR / "training_report_xgb_retrained.json")

        # Print run information
        print_run_info()

    print("\nRetraining completed successfully!")


if __name__ == "__main__":
    sys.exit(main())

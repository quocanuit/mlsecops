import os
try:
    import mlflow
    import mlflow.sklearn
    import mlflow.xgboost
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False


def is_mlflow_enabled():
    """Check if MLflow tracking is enabled via environment variable"""
    return bool(MLFLOW_AVAILABLE and os.getenv("MLFLOW_TRACKING_URI"))


def setup_mlflow(tracking_uri=None, experiment_name="mlsecops-fraud-detection"):
    if not MLFLOW_AVAILABLE:
        return False

    # Get tracking URI from env or parameter
    tracking_uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI")

    if not tracking_uri:
        return False

    try:
        mlflow.set_tracking_uri(tracking_uri)
        print(f"[MLflow] Enabled - Tracking URI: {tracking_uri}")

        # Get or create experiment
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            mlflow.create_experiment(experiment_name)
            print(f"[MLflow] Created experiment: {experiment_name}")
        else:
            print(f"[MLflow] Using experiment: {experiment_name}")

        mlflow.set_experiment(experiment_name)
        return True

    except Exception as e:
        print(f"[MLflow] Warning: Could not connect - {e}")
        print(f"[MLflow] Continuing without MLflow tracking...")
        return False


def log_evaluation_metrics(
    roc_auc, logloss, train_time, prediction_time, total_time,
    accuracy_default, prec_default, rec_default, f1_default, fpr_default, tpr_default,
    fixed_threshold, accuracy_fixed, prec_fixed, rec_fixed, f1_fixed, fpr_at, tpr_at
):
    if not is_mlflow_enabled():
        return

    try:
        # Log common metrics
        mlflow.log_metric("roc_auc", roc_auc)
        mlflow.log_metric("log_loss", logloss)
        mlflow.log_metric("train_time_seconds", train_time)
        mlflow.log_metric("prediction_time_seconds", prediction_time)
        mlflow.log_metric("total_time_seconds", total_time)

        # Log default threshold metrics
        mlflow.log_metric("default_threshold", 0.5)
        mlflow.log_metric("default_accuracy", accuracy_default)
        mlflow.log_metric("default_precision", prec_default)
        mlflow.log_metric("default_recall", rec_default)
        mlflow.log_metric("default_f1", f1_default)
        mlflow.log_metric("default_fpr", fpr_default)
        mlflow.log_metric("default_tpr", tpr_default)

        # Log fixed threshold metrics
        mlflow.log_metric("fixed_threshold", fixed_threshold)
        mlflow.log_metric("fixed_accuracy", accuracy_fixed)
        mlflow.log_metric("fixed_precision", prec_fixed)
        mlflow.log_metric("fixed_recall", rec_fixed)
        mlflow.log_metric("fixed_f1", f1_fixed)
        mlflow.log_metric("fixed_fpr", fpr_at)
        mlflow.log_metric("fixed_tpr", tpr_at)

        print("[MLflow] Logged evaluation metrics")
    except Exception as e:
        print(f"[MLflow] Warning: Could not log metrics - {e}")


def log_model_and_params(model, model_name, metadata, model_path,
                          mlflow_model_name: str = "model",
                          signature=None, input_example=None):
    if not is_mlflow_enabled():
        return

    try:
        # Log model parameters
        mlflow.log_params(metadata["model_params"])

        # Log dataset info
        mlflow.log_param("n_train", metadata["n_train"])
        mlflow.log_param("n_test", metadata["n_test"])
        mlflow.log_param("n_features", metadata["n_features"])
        mlflow.log_param("random_state", metadata["random_state"])
        mlflow.log_param("target_fpr", metadata["fixed_fpr"])

        # Log package versions
        mlflow.log_param("sklearn_version", metadata["sklearn_version"])
        mlflow.log_param("xgboost_version", metadata["xgboost_version"])

        # Log the models
        if "XGBoost" in model_name or "xgb" in str(type(model)).lower():
            mlflow.xgboost.log_model(
                model,
                name=mlflow_model_name,
                signature=signature,
                input_example=input_example
            )
        else:
            mlflow.sklearn.log_model(
                model,
                name=mlflow_model_name,
                signature=signature,
                input_example=input_example
            )

        # Log the complete bundle as artifact
        mlflow.log_artifact(str(model_path), "model_bundle")

        print(f"[MLflow] Logged model and parameters")
    except Exception as e:
        print(f"[MLflow] Warning: Could not log model - {e}")


def log_training_report(report_path):
    if not is_mlflow_enabled():
        return

    try:
        mlflow.log_artifact(str(report_path), "reports")
        print(f"[MLflow] Logged training report")
    except Exception as e:
        print(f"[MLflow] Warning: Could not log report - {e}")


def start_run(run_name=None):
    """Start MLflow run if enabled, otherwise return dummy context"""
    if is_mlflow_enabled() and MLFLOW_AVAILABLE:
        try:
            return mlflow.start_run(run_name=run_name)
        except Exception as e:
            print(f"[MLflow] Warning: Could not start run - {e}")
            print(f"[MLflow] Continuing without tracking...")
            from contextlib import nullcontext
            return nullcontext()
    else:
        from contextlib import nullcontext
        return nullcontext()


def set_tag(key, value):
    """Set MLflow tag if enabled"""
    if is_mlflow_enabled():
        try:
            mlflow.set_tag(key, value)
        except Exception as e:
            print(f"[MLflow] Warning: Could not set tag - {e}")


def print_run_info():
    if not is_mlflow_enabled():
        return

    try:
        active_run = mlflow.active_run()
        if active_run:
            run_id = active_run.info.run_id
            experiment_id = active_run.info.experiment_id
            tracking_uri = mlflow.get_tracking_uri()
            print(f"\n[MLflow] Run ID: {run_id}")
            print(f"[MLflow] View at: {tracking_uri}/#/experiments/{experiment_id}/runs/{run_id}")
    except Exception as e:
        print(f"[MLflow] Warning: Could not get run info - {e}")

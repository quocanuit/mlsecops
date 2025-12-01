import os
import hashlib
try:
    import mlflow
    import mlflow.sklearn
    import mlflow.xgboost
    from mlflow.data.pandas_dataset import PandasDataset
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False


def is_mlflow_enabled():
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


def log_evaluation_metrics(accuracy, precision, recall, f1, train_time, prediction_time):
    """Log simple evaluation metrics to MLflow"""
    if not is_mlflow_enabled():
        return

    try:
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("train_time_seconds", train_time)
        mlflow.log_metric("prediction_time_seconds", prediction_time)
        mlflow.log_metric("total_time_seconds", train_time + prediction_time)

        print("[MLflow] Logged evaluation metrics")
    except Exception as e:
        print(f"[MLflow] Warning: Could not log metrics - {e}")


def log_model_and_params(model, model_name, model_path,
                          mlflow_model_name: str = "model",
                          signature=None, input_example=None):
    """Log model to MLflow"""
    if not is_mlflow_enabled():
        return

    try:
        # Log model parameters
        mlflow.log_params(model.get_params())

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

        # Log the model file as artifact
        mlflow.log_artifact(str(model_path), "model_artifacts")

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
    if is_mlflow_enabled():
        try:
            mlflow.set_tag(key, value)
        except Exception as e:
            print(f"[MLflow] Warning: Could not set tag - {e}")


def log_dataset(X_train, y_train, X_test, y_test, dataset_name="training_data"):
    if not is_mlflow_enabled():
        return

    try:
        import pandas as pd

        # Create dataframes for logging
        train_df = pd.DataFrame(X_train)
        train_df['target'] = y_train

        test_df = pd.DataFrame(X_test)
        test_df['target'] = y_test

        # Log training dataset
        train_dataset = PandasDataset(
            df=train_df,
            source=f"{dataset_name}_train",
            name=f"{dataset_name}_train"
        )
        mlflow.log_input(train_dataset, context="training")

        # Log test dataset
        test_dataset = PandasDataset(
            df=test_df,
            source=f"{dataset_name}_test",
            name=f"{dataset_name}_test"
        )
        mlflow.log_input(test_dataset, context="evaluation")

        print(f"[MLflow] Logged datasets with {train_df.shape[0]} train and {test_df.shape[0]} test samples")
    except Exception as e:
        print(f"[MLflow] Warning: Could not log dataset - {e}")


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

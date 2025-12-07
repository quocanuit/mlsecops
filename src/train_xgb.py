import sys
import time
from pathlib import Path
import os

import xgboost as xgb
from sklearn.metrics import classification_report
from mlflow.models import infer_signature

from utils import train_common, mlflow_common

ROOT = Path(os.getcwd())


def main():
    print("XGBOOST MODEL TRAINING")

    mlflow_common.setup_mlflow()

    # Start MLflow run (no-op if MLflow disabled)
    with mlflow_common.start_run(run_name="xgb_training"):
        mlflow_common.set_tag("model_type", "XGBoost")
        mlflow_common.set_tag("algorithm", "xgboost.XGBClassifier")

        # Load preprocessed data
        df_preprocessed = train_common.load_preprocessed_data()

        # Split data
        X_train, X_test, y_train, y_test = train_common.split_data(df_preprocessed)

        # Calculate the scale_pos_weight parameter
        negative_class_count = len(y_train[y_train == 0])
        positive_class_count = len(y_train[y_train == 1])
        scale_pos_weight = negative_class_count / positive_class_count

        # Initialize and train model
        print("\nTraining XGBoost model...")
        model_xgb = XGBClassifier(
            n_estimators=100,
            eval_metric='logloss',
            scale_pos_weight=scale_pos_weight,
            random_state=42
        )

        start_time = time.time()
        model_xgb.fit(X_train, y_train)
        train_time = time.time() - start_time

        # Predict on the test set
        start_time = time.time()
        y_pred = model_xgb.predict(X_test)
        y_proba = model_xgb.predict_proba(X_test)[:, 1]
        prediction_time = time.time() - start_time

        # Evaluate and get metrics
        print("\nEvaluation Metrics:")
        metrics = train_common.print_metrics(y_test, y_pred, y_proba, train_time, prediction_time)

        # Create metadata
        metadata = train_common.create_metadata(X_train, X_test, model_xgb)

        # Get classification report
        clf_report = classification_report(y_test, y_pred, output_dict=True)

        # Log metrics to MLflow
        mlflow_common.log_evaluation_metrics(metrics)

        # Save model bundle
        model_path = train_common.save_model(model_xgb, "XGBoost Classifier", metadata, "xgb_model_bundle.pkl")

        try:
            sig = infer_signature(
                X_train[:100],
                model_xgb.predict_proba(X_train[:100])[:, 1]
            )
        except Exception:
            sig = None
        input_example = X_train[:5]

        # Log model to MLflow
        mlflow_common.log_model_and_params(
            model_xgb, "XGBoost Classifier", metadata, model_path,
            mlflow_model_name="xgb_model",
            signature=sig, input_example=input_example
        )

        # Save training report
        train_common.save_training_report(metrics, clf_report, metadata, "training_report_xgb.json")

        # Log report to MLflow
        mlflow_common.log_training_report(train_common.ARTIFACTS_DIR / "training_report_xgb.json")

        # Print run information
        mlflow_common.print_run_info()

    print("\nTraining completed!")


if __name__ == "__main__":
    sys.exit(main())

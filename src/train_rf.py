import sys
import time
from pathlib import Path
import os

from sklearn.metrics import classification_report
from imblearn.ensemble import BalancedRandomForestClassifier
from mlflow.models import infer_signature

from utils import train_common, mlflow_common

ROOT = Path(os.getcwd())


def main():
    """Main execution function."""
    print("BALANCED RANDOM FOREST MODEL TRAINING")

    mlflow_common.setup_mlflow()

    # Start MLflow run (no-op if MLflow disabled)
    with mlflow_common.start_run(run_name="brf_training"):
        mlflow_common.set_tag("model_type", "Balanced Random Forest")
        mlflow_common.set_tag("algorithm", "imblearn.ensemble.BalancedRandomForestClassifier")

        # Load preprocessed data
        df_preprocessed = train_common.load_preprocessed_data()

        # Split data
        X_train, X_test, y_train, y_test = train_common.split_data(df_preprocessed)

        # Initialize and train model
        print("\nTraining Balanced Random Forest model...")
        model_brf = BalancedRandomForestClassifier(
            n_estimators=100,
            sampling_strategy='auto',
            max_depth=None,
            random_state=42
        )

        start_time = time.time()
        model_brf.fit(X_train, y_train)
        train_time = time.time() - start_time

        # Predict on the test set
        start_time = time.time()
        y_pred = model_brf.predict(X_test)
        y_proba = model_brf.predict_proba(X_test)[:, 1]
        prediction_time = time.time() - start_time

        # Evaluate and get metrics
        print("\nEvaluation Metrics:")
        metrics = train_common.print_metrics(y_test, y_pred, y_proba, train_time, prediction_time)

        # Create metadata
        metadata = train_common.create_metadata(X_train, X_test, model_brf)
        
        # Get classification report
        clf_report = classification_report(y_test, y_pred, output_dict=True)

        # Log metrics to MLflow
        mlflow_common.log_evaluation_metrics(metrics)

        # Save model bundle
        model_path = train_common.save_model(model_brf, "Balanced Random Forest Classifier", metadata, "brf_model_bundle.pkl")

        try:
            sig = infer_signature(
                X_train[:100],
                model_brf.predict_proba(X_train[:100])[:, 1]
            )
        except Exception:
            sig = None
        input_example = X_train[:5]

        # Log model to MLflow
        mlflow_common.log_model_and_params(
            model_brf, "Balanced Random Forest Classifier", metadata, model_path,
            mlflow_model_name="brf_model",
            signature=sig, input_example=input_example
        )

        # Save training report
        train_common.save_training_report(metrics, clf_report, metadata, "training_report_brf.json")

        # Log report to MLflow
        mlflow_common.log_training_report(train_common.ARTIFACTS_DIR / "training_report_brf.json")

        # Print run information
        mlflow_common.print_run_info()

    print("\nTraining completed successfully!")


if __name__ == "__main__":
    sys.exit(main())

import sys
import time
from pathlib import Path
import os

from sklearn.ensemble import RandomForestClassifier

from utils.train_common import (
    load_preprocessed_data,
    evaluate_model,
    save_model,
    save_training_report
)
from utils.mlflow_common import (
    setup_mlflow,
    start_run,
    set_tag,
    log_evaluation_metrics,
    log_model_and_params,
    log_training_report,
    log_dataset,
    print_run_info
)

ROOT = Path(os.getcwd())
RANDOM_STATE_VALUE = 42


def train_model(X_train, y_train, X_test):
    """Train Random Forest with simple config like just-like-this.py"""
    model = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE_VALUE)

    print("Training Random Forest model...")
    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time

    start_time = time.time()
    y_test_pred = model.predict(X_test)
    prediction_time = time.time() - start_time

    return model, y_test_pred, train_time, prediction_time


def main():
    """Main execution function."""
    print("RANDOM FOREST MODEL TRAINING")

    setup_mlflow()

    with start_run(run_name="random_forest_training"):
        set_tag("model_type", "Random Forest")
        set_tag("algorithm", "sklearn.ensemble.RandomForestClassifier")

        # Load preprocessed data
        X_train_resampled, y_train_resampled, X_test_transformed, y_test = load_preprocessed_data()

        # Train the model
        model, y_test_pred, train_time, prediction_time = train_model(
            X_train_resampled, y_train_resampled, X_test_transformed
        )

        # Evaluate the model
        accuracy, precision, recall, f1, report = evaluate_model(
            "Random Forest", y_test, y_test_pred, train_time, prediction_time
        )
        
        # Save training report
        report_path = save_training_report(
            "Random Forest", accuracy, precision, recall, f1, train_time, prediction_time, report
        )

        # Log metrics to MLflow
        log_evaluation_metrics(accuracy, precision, recall, f1, train_time, prediction_time)

        # Save model
        model_path = save_model(model, "Random Forest Classifier", "rf_model.pkl")

        # Log model to MLflow
        try:
            from mlflow.models import infer_signature
            sig = infer_signature(X_train_resampled[:100], model.predict(X_train_resampled[:100]))
        except Exception:
            sig = None
        
        log_model_and_params(
            model, "Random Forest Classifier", model_path,
            mlflow_model_name="random_forest_model",
            signature=sig, input_example=X_train_resampled[:5]
        )
        
        # Log training report to MLflow
        from utils.mlflow_common import log_training_report
        log_training_report(report_path)

        print_run_info()

    print("\nTraining completed successfully!")


if __name__ == "__main__":
    sys.exit(main())

import sys
import time
from pathlib import Path
import os

from sklearn.ensemble import RandomForestClassifier
from mlflow.models import infer_signature

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


def train_model(X_train, y_train, X_test):
    model = RandomForestClassifier(random_state=RANDOM_STATE_VALUE)

    print("Training Random Forest model...")
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

    return model, y_test_pred, y_test_prob, train_time, prediction_time, metadata


def main():
    """Main execution function."""
    print("RANDOM FOREST MODEL TRAINING")

    setup_mlflow()

    # Start MLflow run (no-op if MLflow disabled)
    with start_run(run_name="random_forest_training"):
        set_tag("model_type", "Random Forest")
        set_tag("algorithm", "sklearn.ensemble.RandomForestClassifier")

        # Load preprocessed data
        X_train_resampled, y_train_resampled, X_test_transformed, y_test = load_preprocessed_data()

        # Train the model
        model, y_test_pred, y_test_prob, train_time, prediction_time, metadata = train_model(
            X_train_resampled, y_train_resampled, X_test_transformed
        )

        # Evaluate the model
        results_default, results_fixed, fixed_threshold, report_default, report_fixed = evaluate_model(
            "Random Forest", y_test, y_test_pred, y_test_prob, train_time, prediction_time, FIXED_FPR
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
        model_path = save_model(model, "Random Forest Classifier", fixed_threshold, FIXED_FPR, metadata, "rf_model_bundle.pkl")

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
            model, "XGBoost Classifier", metadata, model_path,
            mlflow_model_name="random_forest_model",
            signature=sig, input_example=input_example
        )

        # Log model to MLflow
        log_model_and_params(model, "Random Forest Classifier", metadata, model_path)

        # Save training report
        save_training_report(results_default, results_fixed, report_default, report_fixed, metadata, "training_report_rf.json")

        # Log report to MLflow
        log_training_report(ARTIFACTS_DIR / "training_report_rf.json")

        # Print run information
        print_run_info()

    print("\nTraining completed successfully!")


if __name__ == "__main__":
    sys.exit(main())

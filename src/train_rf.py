import sys
import time
from pathlib import Path
import os

from sklearn.ensemble import RandomForestClassifier

from utils.train_common import (
    load_preprocessed_data,
    create_metadata,
    evaluate_model,
    save_model,
    save_training_report
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

    # Save model bundle
    save_model(model, "Random Forest Classifier", fixed_threshold, FIXED_FPR, metadata, "rf_model_bundle.pkl")

    # Save training report
    save_training_report(results_default, results_fixed, report_default, report_fixed, metadata, "training_report_rf.json")

    print("\nTraining completed successfully!")


if __name__ == "__main__":
    sys.exit(main())

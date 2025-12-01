import pandas as pd
import numpy as np
import json
import joblib
from pathlib import Path
import os

from sklearn.metrics import (
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

ROOT = Path(os.getcwd())
ARTIFACTS_DIR = Path(os.getenv("ARTIFACTS_DIR", str(ROOT / "artifacts")))

def load_preprocessed_data():
    legacy_dir = ROOT / "data" / "preprocessed"
    env_dir = os.getenv("PREPROCESSED_DIR", "").strip()
    base_dir = Path(env_dir) if env_dir else legacy_dir

    X_train_resampled = pd.read_csv(base_dir / "X_train_resampled.csv").values
    y_train_resampled = pd.read_csv(base_dir / "y_train_resampled.csv").values.ravel()
    X_test_transformed = pd.read_csv(base_dir / "X_test_transformed.csv").values
    y_test = pd.read_csv(base_dir / "y_test.csv").values.ravel()

    print(f"[load_preprocessed_data] Using dir: {base_dir}")
    print(f"  X_train_resampled: {X_train_resampled.shape}")
    print(f"  y_train_resampled: {y_train_resampled.shape}")
    print(f"  X_test_transformed: {X_test_transformed.shape}")
    print(f"  y_test: {y_test.shape}")

    return X_train_resampled, y_train_resampled, X_test_transformed, y_test


def evaluate_model(model_name, y_test, y_test_pred, train_time, prediction_time):
    """Simple evaluation with basic metrics like just-like-this.py"""
    accuracy = accuracy_score(y_test, y_test_pred)
    precision = precision_score(y_test, y_test_pred, zero_division=0)
    recall = recall_score(y_test, y_test_pred, zero_division=0)
    f1 = f1_score(y_test, y_test_pred, zero_division=0)
    
    print(f"\n{model_name} Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Training time: {train_time:.2f}s")
    print(f"Prediction time: {prediction_time:.2f}s")
    
    report = classification_report(y_test, y_test_pred, output_dict=True)
    
    return accuracy, precision, recall, f1, report


def save_model(model, model_name, bundle_filename):
    """Simple model save without metadata"""
    models_dir = ARTIFACTS_DIR / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = models_dir / bundle_filename
    joblib.dump(model, model_path)
    
    print(f"Model saved to: {model_path}")
    return model_path


def save_training_report(model_name, accuracy, precision, recall, f1, train_time, prediction_time, report):
    """Save simplified training report"""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    
    training_report = {
        "model": model_name,
        "metrics": {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "train_time_seconds": float(train_time),
            "prediction_time_seconds": float(prediction_time)
        },
        "classification_report": report
    }
    
    report_filename = f"training_report_{model_name.lower().replace(' ', '_')}.json"
    report_path = ARTIFACTS_DIR / report_filename
    
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(training_report, f, ensure_ascii=False, indent=2)
    
    print(f"Training report saved to: {report_path}")
    return report_path




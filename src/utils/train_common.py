import pandas as pd
import numpy as np
import json
import joblib
from pathlib import Path
import os
from datetime import datetime
import sklearn
import xgboost
import imblearn

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    log_loss
)

ROOT = Path(os.getcwd())
ARTIFACTS_DIR = Path(os.getenv("ARTIFACTS_DIR", str(ROOT / "artifacts")))


def load_preprocessed_data():
    legacy_dir = ROOT / "data" / "preprocessed"
    env_dir = os.getenv("PREPROCESSED_DIR", "").strip()
    base_dir = Path(env_dir) if env_dir else legacy_dir

    preprocessed_file = base_dir / "data_preprocessed.csv"
    if not preprocessed_file.exists():
        raise FileNotFoundError(f"Preprocessed data not found at: {preprocessed_file}")

    print(f"[load_preprocessed_data] Loading from: {base_dir}")
    df_preprocessed = pd.read_csv(preprocessed_file)
    print(f"  data_preprocessed: {df_preprocessed.shape}")

    return df_preprocessed


def split_data(df_preprocessed):
    from sklearn.model_selection import train_test_split
    
    X = df_preprocessed.drop('fraud_bool', axis=1)
    y = df_preprocessed['fraud_bool']
    
    categorical_features = X.select_dtypes(include=['object', 'category']).columns
    X[categorical_features] = X[categorical_features].astype('category')

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y, shuffle=True)
    
    print(f"Train set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    print(f"Train fraud rate: {y_train.mean():.4f}")
    print(f"Test fraud rate: {y_test.mean():.4f}")
    
    return X_train, X_test, y_train, y_test


def create_metadata(X_train, X_test, model):
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "sklearn_version": sklearn.__version__,
        "xgboost_version": xgboost.__version__,
        "imblearn_version": imblearn.__version__,
        "pandas_version": pd.__version__,
        "numpy_version": np.__version__,
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
        "n_features": int(X_train.shape[1]),
        "model_params": model.get_params()
    }
    return metadata

def print_metrics(y_test, y_pred, y_proba, train_time=None, prediction_time=None):
    from sklearn.metrics import precision_score, recall_score, f1_score
    
    # Compute metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    logloss = log_loss(y_test, y_proba)

    # Print the results
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"AUC-ROC: {roc_auc:.4f}")
    print(f"Log Loss: {logloss:.4f}")
    
    # Build metrics dict
    metrics = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "roc_auc": float(roc_auc),
        "log_loss": float(logloss)
    }
    
    # Add timing metrics if provided
    if train_time is not None:
        metrics["train_time_seconds"] = float(train_time)
    if prediction_time is not None:
        metrics["prediction_time_seconds"] = float(prediction_time)
    if train_time is not None and prediction_time is not None:
        metrics["total_time_seconds"] = float(train_time + prediction_time)
    
    return metrics


def save_model(model, model_name, metadata, bundle_filename):
    models_dir = ARTIFACTS_DIR / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    # Create model bundle with metadata
    model_bundle = {
        "model": model,
        "metadata": metadata
    }

    model_path = models_dir / bundle_filename
    joblib.dump(model_bundle, model_path)

    print(f"\n[INFO] Model bundle saved to: {model_path}")
    print(f"  - Model: {model_name}")
    print(f"\nUsage example:")
    print(f"  bundle = joblib.load('{model_path}')")
    print(f"  model = bundle['model']")
    print(f"  y_prob = model.predict_proba(X_new)[:, 1]")
    print(f"  y_pred = model.predict(X_new)")

    return model_path


def save_training_report(metrics, classification_report_dict, metadata, report_filename):
    artifacts_dir = ARTIFACTS_DIR
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # Save results as JSON with metadata
    training_report = {
        "metadata": metadata,
        "evaluation": {
            "metrics": metrics,
            "classification_report": classification_report_dict
        }
    }

    report_path = artifacts_dir / report_filename
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(training_report, f, ensure_ascii=False, indent=2)

    print("[INFO] TRAINING REPORT")
    print(json.dumps(training_report, ensure_ascii=False, indent=2))
    print(f"Report saved to: {report_path}")

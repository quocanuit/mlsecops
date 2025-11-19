import pandas as pd
import numpy as np
import json
import joblib
from pathlib import Path
import os
from datetime import datetime
import sklearn
import xgboost

from sklearn.metrics import (
    classification_report,
    accuracy_score,
    roc_auc_score,
    roc_curve,
    log_loss,
    mean_squared_error,
    precision_recall_fscore_support
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


def create_metadata(X_train, X_test, model, random_state, fixed_fpr):
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "random_state": random_state,
        "fixed_fpr": fixed_fpr,
        "sklearn_version": sklearn.__version__,
        "xgboost_version": xgboost.__version__,
        "pandas_version": pd.__version__,
        "numpy_version": np.__version__,
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
        "n_features": int(X_train.shape[1]),
        "model_params": model.get_params()
    }
    return metadata


def evaluate_model(model_name, y_test, y_test_pred, y_test_prob, train_time, prediction_time, fixed_fpr):
    # ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_test_prob)

    # common metrics
    roc_auc = roc_auc_score(y_test, y_test_prob)
    logloss = log_loss(y_test, y_test_prob)
    total_time = train_time + prediction_time

    # 1. Default Threshold (0.5)
    accuracy_default = accuracy_score(y_test, y_test_pred)
    mse_default = mean_squared_error(y_test, y_test_pred)

    # Compute precision, recall, F1 for default threshold
    prec_default, rec_default, f1_default, _ = precision_recall_fscore_support(
        y_test, y_test_pred, average="binary", zero_division=0
    )

    idx_default = np.argmin(np.abs(thresholds - 0.5))
    fpr_default = float(fpr[idx_default])
    tpr_default = float(tpr[idx_default])

    results_default = {
        "Model": model_name,
        "Decision Threshold": 0.5,
        "FPR@threshold": fpr_default,
        "TPR@threshold": tpr_default,
        "Accuracy": float(accuracy_default),
        "Precision": float(prec_default),
        "Recall": float(rec_default),
        "F1 Score": float(f1_default),
        "ROC-AUC Score": float(roc_auc),
        "Log Loss": float(logloss),
        "Mean Squared Error": float(mse_default),
        "Training Time (s)": float(train_time),
        "Prediction Time (s)": float(prediction_time),
        "Total Time (s)": float(total_time)
    }

    results_df_default = pd.DataFrame([results_default])

    # 2. Fixed Threshold at Fixed FPR
    # Guard the fixed-FPR indexing
    mask = fpr <= fixed_fpr
    if not np.any(mask):
        # Fallback to the lowest-FPR point
        idx = 0
        print(f"[WARNING] No threshold achieves FPR <= {fixed_fpr*100}%. Using lowest FPR available.")
    else:
        idx = np.nonzero(mask)[0][-1]

    fixed_threshold = float(thresholds[idx])
    fpr_at = float(fpr[idx])
    tpr_at = float(tpr[idx])

    print(f"[INFO] Fixed threshold selected: {fixed_threshold:.6f}")
    print(f"       Achieved FPR: {fpr_at:.6f} (target: {fixed_fpr})")
    print(f"       Achieved TPR: {tpr_at:.6f}")

    # Predictions at fixed threshold
    y_pred_fixed = (y_test_prob >= fixed_threshold).astype(int)

    accuracy_fixed = accuracy_score(y_test, y_pred_fixed)
    mse_fixed = mean_squared_error(y_test, y_pred_fixed)

    # Compute precision, recall, F1 for fixed threshold
    prec_fixed, rec_fixed, f1_fixed, _ = precision_recall_fscore_support(
        y_test, y_pred_fixed, average="binary", zero_division=0
    )

    results_fixed = {
        "Model": model_name,
        "Decision Threshold": float(fixed_threshold),
        "Target FPR": float(fixed_fpr),
        "FPR@threshold": fpr_at,
        "TPR@threshold": tpr_at,
        "Accuracy": float(accuracy_fixed),
        "Precision@threshold": float(prec_fixed),
        "Recall@threshold": float(rec_fixed),
        "F1@threshold": float(f1_fixed),
        "ROC-AUC Score": float(roc_auc),
        "Log Loss": float(logloss),
        "Mean Squared Error": float(mse_fixed),
        "Training Time (s)": float(train_time),
        "Prediction Time (s)": float(prediction_time),
        "Total Time (s)": float(total_time)
    }

    results_df_fixed = pd.DataFrame([results_fixed])

    # Get classification reports
    report_default = classification_report(y_test, y_test_pred, output_dict=True)
    report_fixed = classification_report(y_test, y_pred_fixed, output_dict=True)

    return results_df_default, results_df_fixed, fixed_threshold, report_default, report_fixed


def save_model(model, model_name, fixed_threshold, fixed_fpr, metadata, bundle_filename):
    models_dir = ARTIFACTS_DIR / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    # Create model bundle with metadata
    model_bundle = {
        "model": model,
        "fixed_threshold": fixed_threshold,
        "fixed_fpr": fixed_fpr,
        "metadata": metadata
    }

    model_path = models_dir / bundle_filename
    joblib.dump(model_bundle, model_path)

    print(f"\n[INFO] Model bundle saved to: {model_path}")
    print(f"  - Model: {model_name}")
    print(f"  - Fixed Threshold: {fixed_threshold:.4f}")
    print(f"  - Target FPR: {fixed_fpr*100}%")
    print(f"\nUsage example:")
    print(f"  bundle = joblib.load('{model_path}')")
    print(f"  model = bundle['model']")
    print(f"  threshold = bundle['fixed_threshold']")
    print(f"  y_prob = model.predict_proba(X_new)[:, 1]")
    print(f"  y_pred = (y_prob >= threshold).astype(int)")

    return model_path


def save_training_report(results_default, results_fixed, report_default, report_fixed, metadata, report_filename):
    artifacts_dir = ARTIFACTS_DIR
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # Save results as JSON with metadata
    training_report = {
        "metadata": metadata,
        "default_threshold_evaluation": {
            "metrics": results_default.to_dict(orient='records')[0],
            "classification_report": report_default
        },
        "fixed_threshold_evaluation": {
            "metrics": results_fixed.to_dict(orient='records')[0],
            "classification_report": report_fixed
        }
    }

    report_path = artifacts_dir / report_filename
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(training_report, f, ensure_ascii=False, indent=2)

    print("[INFO] TRAINING REPORT")
    print(json.dumps(training_report, ensure_ascii=False, indent=2))
    print(f"Report saved to: {report_path}")

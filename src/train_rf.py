import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    auc,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)

ROOT = Path(os.getcwd())
print(f"Project ROOT: {ROOT}")
def load_data():
    data_path = ROOT / "ICAIF_KAGGLE" / "extracted"

    X_train = pd.read_csv(data_path / "X_train.csv")
    X_test = pd.read_csv(data_path / "X_test.csv")
    y_train = pd.read_csv(data_path / "y_train.csv").squeeze()
    y_test = pd.read_csv(data_path / "y_test.csv").squeeze()

    print("Train size:", X_train.shape)
    print("Test size:", X_test.shape)
    print(f"Total features: {X_train.shape[1]}")

    return X_train, X_test, y_train, y_test


def create_model_pipeline():
    rf = RandomForestClassifier(
        n_estimators=100,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    )

    pipeline = ImbPipeline(steps=[
        ("smote", SMOTE(random_state=42)),
        ("rf", rf),
    ])

    return pipeline


def evaluate_model(pipeline, X_test, y_test):
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    # Calculate metrics
    roc = roc_auc_score(y_test, y_prob)
    prec, rec, thr = precision_recall_curve(y_test, y_prob)
    pr_auc = auc(rec, prec)
    cm = confusion_matrix(y_test, y_pred)

    # Print results
    print("=== Default-Threshold Metrics (0.5) ===")
    print(f"ROC-AUC : {roc:.4f}")
    print(f"PR-AUC  : {pr_auc:.4f}")
    print(classification_report(y_test, y_pred, digits=4))
    print("Confusion Matrix:\n", cm)

    return roc, pr_auc, y_pred


def save_model_and_report(pipeline, X_train, X_test, y_train, y_test, roc, pr_auc):
    # Save model
    models_dir = ROOT / "artifacts" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / "rf_smote_pipeline.pkl"
    joblib.dump(pipeline, model_path)

    # Create training report
    report = {
        "model_type": "RandomForestClassifier_with_SMOTE",
        "preprocessing": "Done in extract_features.py",
        "n_estimators": 100,
        "min_samples_split": 5,
        "train_samples": int(X_train.shape[0]),
        "test_samples": int(X_test.shape[0]),
        "n_features": int(X_train.shape[1]),
        "roc_auc": float(roc),
        "pr_auc": float(pr_auc),
        "fraud_rate_train": float(y_train.mean()),
        "fraud_rate_test": float(y_test.mean()),
        "model_path": str(model_path.relative_to(ROOT)),
    }

    # Save training report
    report_path = models_dir.parent / "train_rf_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n=== TRAINING REPORT ===")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Model saved to: {model_path}")
    print(f"Saved report -> {report_path}")


def main():
    X_train, X_test, y_train, y_test = load_data()
    pipeline = create_model_pipeline()
    pipeline.fit(X_train, y_train)

    roc, pr_auc, y_pred = evaluate_model(pipeline, X_test, y_test)

    save_model_and_report(pipeline, X_train, X_test, y_train, y_test, roc, pr_auc)


if __name__ == "__main__":
    main()

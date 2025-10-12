from pathlib import Path
import os
import json

ROOT = Path(os.getcwd())
print(f"Project ROOT: {ROOT}")
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
df = pd.read_csv(f"{ROOT}/ICAIF_KAGGLE/raw/train.csv")
# Target column
y = df["fraud_bool"]

# Features (drop target + non-informative ID column)
X = df.drop(["fraud_bool", "case_id"], axis=1)
# Handle categorical variables (label encoding for simplicity)
categorical_cols = X.select_dtypes(include=["object"]).columns

le = LabelEncoder()
for col in categorical_cols:
    X[col] = le.fit_transform(X[col].astype(str))

print("Features after encoding:", X.shape)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Train size:", X_train.shape)
print("Test size:", X_test.shape)
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced",
    verbose=1
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
import joblib

# Save model using ROOT/artifacts/models path
models_dir = ROOT / "artifacts" / "models"
models_dir.mkdir(parents=True, exist_ok=True)
model_path = models_dir / "rf.pkl"
joblib.dump(model, model_path)

# Create training report
report = {
    "model_type": "RandomForestClassifier",
    "n_estimators": 100,
    "train_samples": int(X_train.shape[0]),
    "test_samples": int(X_test.shape[0]),
    "n_features": int(X_train.shape[1]),
    "accuracy": float(accuracy_score(y_test, y_pred)),
    "fraud_rate_train": float(y_train.mean()),
    "fraud_rate_test": float(y_test.mean()),
    "model_path": str(model_path.relative_to(ROOT)),
}

# Save training report
report_path = models_dir.parent / "train_rf_report.json"

with open(report_path, "w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=2)

print("=== TRAINING REPORT ===")
print(json.dumps(report, ensure_ascii=False, indent=2))
print(f"Model saved to: {model_path}")
print(f"Saved report -> {report_path}")
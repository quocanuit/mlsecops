import pandas as pd
import json
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import NearMiss
from pathlib import Path
import numpy as np
import sys
import os

ROOT = Path(os.getcwd())
VALIDATED_DIR = Path(os.getenv("VALIDATED_DIR", str(ROOT / "data" / "validated")))
PREPROCESSED_DIR = Path(os.getenv("PREPROCESSED_DIR", str(ROOT / "artifacts" / "preprocessed")))
RANDOM_STATE_VALUE = 42


def preprocess_data():
    # Load data
    dataset = pd.read_csv(VALIDATED_DIR / "Base_validated.csv")
    print("Original shape:", dataset.shape)
    
    # Drop unnecessary columns
    dataset = dataset.drop(["device_fraud_count", "month"], axis=1)
    
    # OneHot encoding
    dataset_dummy = pd.get_dummies(dataset, drop_first=True)
    
    X = dataset_dummy.drop(["fraud_bool"], axis=1)
    y = dataset_dummy['fraud_bool']
    
    # Apply NearMiss undersampling
    print("Applying NearMiss undersampling...")
    nm = NearMiss()
    X_res, y_res = nm.fit_resample(X, y)
    print(f"After NearMiss: {X_res.shape}, fraud rate: {y_res.mean():.3f}")
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_res, y_res, test_size=0.2, random_state=RANDOM_STATE_VALUE
    )
    
    # Scale numerical features with MinMaxScaler
    numerical_cols = X_train.select_dtypes(include=np.number).columns
    scaler = MinMaxScaler()
    scaler.fit(X_train[numerical_cols])
    X_train[numerical_cols] = scaler.transform(X_train[numerical_cols])
    X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])
    
    print("Train size:", X_train.shape[0])
    print("Test size:", X_test.shape[0])
    
    # Generate simple report
    report = {
        "method": "NearMiss + MinMaxScaler",
        "random_state": RANDOM_STATE_VALUE,
        "train_samples": int(X_train.shape[0]),
        "test_samples": int(X_test.shape[0]),
        "n_features": int(X_train.shape[1]),
        "train_fraud_rate": float(y_train.mean()),
        "test_fraud_rate": float(y_test.mean())
    }
    
    return X_train.values, y_train.values, X_test.values, y_test.values, report





def save_preprocessed_data(X_train, y_train, X_test, y_test):
    PREPROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    
    n_features = X_train.shape[1]
    feature_names = [f"feature_{i}" for i in range(n_features)]
    
    pd.DataFrame(X_train, columns=feature_names).to_csv(PREPROCESSED_DIR / "X_train_resampled.csv", index=False)
    pd.DataFrame(y_train, columns=['fraud_bool']).to_csv(PREPROCESSED_DIR / "y_train_resampled.csv", index=False)
    pd.DataFrame(X_test, columns=feature_names).to_csv(PREPROCESSED_DIR / "X_test_transformed.csv", index=False)
    pd.DataFrame(y_test, columns=['fraud_bool']).to_csv(PREPROCESSED_DIR / "y_test.csv", index=False)
    
    print(f"Saved to {PREPROCESSED_DIR}: X_train {X_train.shape}, X_test {X_test.shape}")


def save_report(report):
    """Save preprocessing report"""
    artifacts_dir = Path(os.getenv("ARTIFACTS_DIR", str(ROOT / "artifacts")))
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    
    report_path = artifacts_dir / "preprocessing_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print(f"Preprocessing report saved to: {report_path}")





def main():
    X_train, y_train, X_test, y_test, report = preprocess_data()
    save_preprocessed_data(X_train, y_train, X_test, y_test)
    save_report(report)
    print("\nPreprocessing completed!")


if __name__ == "__main__":
    sys.exit(main())

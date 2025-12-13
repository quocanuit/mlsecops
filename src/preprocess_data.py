import pandas as pd
import numpy as np
import json
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder
from pathlib import Path
import sys
import os

ROOT = Path(os.getcwd())
ARTIFACTS_DIR = Path(os.getenv("ARTIFACTS_DIR", str(ROOT / "artifacts")))
VALIDATED_DIR = Path(os.getenv("VALIDATED_DIR", str(ROOT / "data" / "validated")))
PREPROCESSED_DIR = Path(os.getenv("PREPROCESSED_DIR", str(ROOT / "artifacts" / "preprocessed")))


def preprocess_with_labelencoder(df: pd.DataFrame, col_label: str):
    # Identify categorical and numerical features
    categorical_features = df.select_dtypes(include=["object", "category"]).columns
    numerical_features = df.select_dtypes(include=["number"]).columns

    categorical_features = [
        features for features in categorical_features if features != col_label
    ]
    numerical_features = [
        features for features in numerical_features if features != col_label
    ]

    # Initialize dictionaries to store the encoders and scaler
    label_encoders = {}
    scaler = StandardScaler()

    # Encode categorical features using LabelEncoder
    for col in categorical_features:
        label_encoders[col] = LabelEncoder()
        df[col] = label_encoders[col].fit_transform(df[col])

    # Scale numerical features
    df[numerical_features] = scaler.fit_transform(df[numerical_features])

    return df, label_encoders, scaler



def generate_report(df_original, df_cleaned, df_preprocessed, label_encoders, numerical_features):
    categorical_features = list(label_encoders.keys())

    report = {
        "method": "LabelEncoder + StandardScaler",
        "data": {
            "original_rows": int(df_original.shape[0]),
            "cleaned_rows": int(df_cleaned.shape[0]),
            "rows_dropped": int(df_original.shape[0] - df_cleaned.shape[0]),
            "fraud_rate": round(float(df_preprocessed['fraud_bool'].mean()), 4)
        },
        "features": {
            "categorical": len(categorical_features),
            "numerical": len(numerical_features),
            "total": len(categorical_features) + len(numerical_features)
        }
    }

    return report


def save_preprocessed_data(df_preprocessed, label_encoders, scaler):
    # Create preprocessed_data directory
    preprocessed_data_dir = PREPROCESSED_DIR
    preprocessed_data_dir.mkdir(parents=True, exist_ok=True)

    # Save preprocessed dataframe
    df_preprocessed.to_csv(preprocessed_data_dir / "data_preprocessed.csv", index=False)

    # Save label encoders
    with open(preprocessed_data_dir / "label_encoders.pkl", "wb") as f:
        pickle.dump(label_encoders, f)

    # Save scaler
    with open(preprocessed_data_dir / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    print("[INFO] PREPROCESSED DATA SAVED")
    print(f"Directory: {preprocessed_data_dir}")
    print(f"Files created:")
    print(f"  - data_preprocessed.csv ({df_preprocessed.shape[0]} rows, {df_preprocessed.shape[1]} columns)")
    print(f"  - label_encoders.pkl (encoders for {len(label_encoders)} categorical features)")
    print(f"  - scaler.pkl (StandardScaler for numerical features)")

    return preprocessed_data_dir


def save_report(report):
    # Create artifacts directory
    artifacts_dir = ARTIFACTS_DIR
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # Save report as JSON
    report_path = artifacts_dir / "preprocessing_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("[INFO] PREPROCESSING REPORT")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Report saved to: {report_path}")


def main():
    df = pd.read_csv(VALIDATED_DIR / "Base_validated.csv")
    print(f"[INFO] Loaded {df.shape[0]:,} rows, {df.shape[1]} columns")

    # Store original row count before modifications
    original_rows = df.shape[0]

    # Drop columns - drop() returns new DataFrame, no need for copy()
    df_cleaned = df.drop(columns=[
        "bank_months_count",
        "prev_address_months_count",
        "velocity_4w"
        ]
    )

    cols_missing = [
    'current_address_months_count',
    'session_length_in_minutes',
    'device_distinct_emails_8w',
    'intended_balcon_amount'
    ]

    print("[INFO] Replacing -1 with NaN in missing value columns...")
    df_cleaned[cols_missing] = df_cleaned[cols_missing].replace(-1, np.nan)

    rows_before_dropna = df_cleaned.shape[0]
    df_cleaned = df_cleaned.dropna()
    print(f"[INFO] Dropped {rows_before_dropna - df_cleaned.shape[0]:,} rows with NaN")
    print(f"[INFO] Remaining: {df_cleaned.shape[0]:,} rows")

    print("[INFO] Starting preprocessing with LabelEncoder and StandardScaler...")
    df_preprocessed, label_encoders, scaler = preprocess_with_labelencoder(df=df_cleaned, col_label="fraud_bool")

    # Get numerical features for report
    numerical_features = df_cleaned.select_dtypes(include=["number"]).columns.tolist()
    numerical_features = [f for f in numerical_features if f != "fraud_bool"]

    # Generate report using stored original_rows count
    # Create minimal df_original placeholder to match function signature
    df_original_placeholder = pd.DataFrame({'_placeholder': range(original_rows)})
    report = generate_report(
        df_original=df_original_placeholder,
        df_cleaned=df_cleaned,
        df_preprocessed=df_preprocessed,
        label_encoders=label_encoders,
        numerical_features=numerical_features
    )
    print("[INFO] Preprocessing completed successfully!")

    # Save preprocessed data
    data_dir = save_preprocessed_data(df_preprocessed, label_encoders, scaler)

    # Save report
    save_report(report)

    print("\nPreprocessing completed successfully!")
    print(f"All data saved to: {data_dir}")


if __name__ == "__main__":
    sys.exit(main())

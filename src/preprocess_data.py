import pandas as pd
import json
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE
from pathlib import Path
import sys
import os

ROOT = Path(os.getcwd())
ARTIFACTS_DIR = Path(os.getenv("ARTIFACTS_DIR", str(ROOT / "artifacts")))
RANDOM_STATE_VALUE = 42


def preprocess_data():
    # Load data
    df_original = pd.read_csv(f"{ROOT}/data/validated/Base_validated.csv")
    print("df.shape:", df_original.shape)

    # Balance classes using downsampling
    df_fraud = df_original[df_original['fraud_bool'] == 1]
    df_non_fraud = df_original[df_original['fraud_bool'] == 0]

    df_non_fraud_downsampled = resample(df_non_fraud,
                                        replace=False,
                                        n_samples=len(df_fraud),
                                        random_state=RANDOM_STATE_VALUE)

    df_balanced = pd.concat([df_fraud, df_non_fraud_downsampled])
    df_balanced = df_balanced.sample(frac=1, random_state=RANDOM_STATE_VALUE).reset_index(drop=True)

    # Define features
    categorical_features = ['payment_type', 'employment_status', 'housing_status', 'source', 'device_os']
    numerical_features = df_balanced.drop(columns=['fraud_bool', 'month'] + categorical_features).columns.tolist()

    # Create preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(sparse_output=False, drop='first'), categorical_features)
        ]
    )

    # Split data by month
    train_data = df_balanced[df_balanced['month'].between(0, 5)]
    test_data = df_balanced[df_balanced['month'].between(6, 7)]

    train_data = train_data.drop(columns=['month'])
    test_data = test_data.drop(columns=['month'])

    train_data.reset_index(drop=True, inplace=True)
    test_data.reset_index(drop=True, inplace=True)

    # Separate features and target
    X_train = train_data.drop('fraud_bool', axis=1)
    y_train = train_data['fraud_bool']
    X_test = test_data.drop('fraud_bool', axis=1)
    y_test = test_data['fraud_bool']

    # Transform features
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    # Apply SMOTE
    smote = SMOTE(random_state=RANDOM_STATE_VALUE)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_transformed, y_train)

    print("Train size:", X_train_resampled.shape[0])
    print("Test size:", X_test_transformed.shape[0])

    # Generate preprocessing report
    report = generate_report(
        df_original=df_original,
        df_balanced=df_balanced,
        X_train_original=X_train_transformed,
        y_train_original=y_train,
        X_train_resampled=X_train_resampled,
        y_train_resampled=y_train_resampled,
        X_test_transformed=X_test_transformed,
        y_test=y_test,
        categorical_features=categorical_features,
        numerical_features=numerical_features
    )

    return X_train_resampled, y_train_resampled, X_test_transformed, y_test, report


def generate_report(df_original, df_balanced, X_train_original, y_train_original,
                                   X_train_resampled, y_train_resampled, X_test_transformed,
                                   y_test, categorical_features, numerical_features):

    report = {
        "configurations": {
            "random_state": RANDOM_STATE_VALUE,
            "train_months": "0-5",
            "test_months": "6-7",
            "balance_method": "downsampling",
            "oversampling_method": "SMOTE",
            "scaler": "StandardScaler",
            "encoder": "OneHotEncoder (drop_first=True)"
        },
        "original_data": {
            "total_rows": int(df_original.shape[0]),
            "total_columns": int(df_original.shape[1]),
            "fraud_count": int(df_original['fraud_bool'].sum()),
            "non_fraud_count": int((df_original['fraud_bool'] == 0).sum()),
            "fraud_rate": float(df_original['fraud_bool'].mean())
        },
        "balanced_data": {
            "total_rows": int(df_balanced.shape[0]),
            "fraud_count": int(df_balanced['fraud_bool'].sum()),
            "non_fraud_count": int((df_balanced['fraud_bool'] == 0).sum()),
            "fraud_rate": float(df_balanced['fraud_bool'].mean()),
            "downsampling_ratio": float(df_balanced.shape[0] / df_original.shape[0])
        },
        "features": {
            "categorical_features": categorical_features,
            "categorical_count": len(categorical_features),
            "numerical_features": numerical_features,
            "numerical_count": len(numerical_features),
            "total_features": len(categorical_features) + len(numerical_features)
        },
        "train_data_before_smote": {
            "samples": int(X_train_original.shape[0]),
            "fraud_count": int(y_train_original.sum()),
            "non_fraud_count": int((y_train_original == 0).sum()),
            "fraud_rate": float(y_train_original.mean())
        },
        "train_data_after_smote": {
            "samples": int(X_train_resampled.shape[0]),
            "features": int(X_train_resampled.shape[1]),
            "fraud_count": int(y_train_resampled.sum()),
            "non_fraud_count": int((y_train_resampled == 0).sum()),
            "fraud_rate": float(y_train_resampled.mean()),
            "smote_increase_ratio": float(X_train_resampled.shape[0] / X_train_original.shape[0])
        },
        "test_data": {
            "samples": int(X_test_transformed.shape[0]),
            "features": int(X_test_transformed.shape[1]),
            "fraud_count": int(y_test.sum()),
            "non_fraud_count": int((y_test == 0).sum()),
            "fraud_rate": float(y_test.mean())
        },
        "data_split": {
            "train_test_ratio": float(X_train_resampled.shape[0] / (X_train_resampled.shape[0] + X_test_transformed.shape[0])),
            "train_percentage": float(X_train_resampled.shape[0] / (X_train_resampled.shape[0] + X_test_transformed.shape[0]) * 100),
            "test_percentage": float(X_test_transformed.shape[0] / (X_train_resampled.shape[0] + X_test_transformed.shape[0]) * 100)
        }
    }

    return report


def save_preprocessed_data(X_train, y_train, X_test, y_test):
    # Create preprocessed_data directory
    preprocessed_data_dir = ROOT / "artifacts" / "preprocessed"
    preprocessed_data_dir.mkdir(parents=True, exist_ok=True)

    n_features = X_train.shape[1]
    feature_names = [f"feature_{i}" for i in range(n_features)]

    # Save training data
    X_train_df = pd.DataFrame(X_train, columns=feature_names)
    y_train_df = pd.DataFrame(y_train, columns=['fraud_bool'])

    X_train_df.to_csv(preprocessed_data_dir / "X_train_resampled.csv", index=False)
    y_train_df.to_csv(preprocessed_data_dir / "y_train_resampled.csv", index=False)

    # Save test data
    X_test_df = pd.DataFrame(X_test, columns=feature_names)
    y_test_df = pd.DataFrame(y_test, columns=['fraud_bool'])

    X_test_df.to_csv(preprocessed_data_dir / "X_test_transformed.csv", index=False)
    y_test_df.to_csv(preprocessed_data_dir / "y_test.csv", index=False)

    print("[INFO] PREPROCESSED DATA SAVED")
    print(f"Directory: {preprocessed_data_dir}")
    print(f"Files created:")
    print(f"  - X_train_resampled.csv ({X_train_df.shape[0]} rows, {X_train_df.shape[1]} features)")
    print(f"  - y_train_resampled.csv ({y_train_df.shape[0]} rows)")
    print(f"  - X_test_transformed.csv ({X_test_df.shape[0]} rows, {X_test_df.shape[1]} features)")
    print(f"  - y_test.csv ({y_test_df.shape[0]} rows)")

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
    X_train, y_train, X_test, y_test, report = preprocess_data()

    # Save preprocessed data
    data_dir = save_preprocessed_data(X_train, y_train, X_test, y_test)

    save_report(report)

    print("\nPreprocessing completed successfully!")
    print(f"Final training set shape: {X_train.shape}")
    print(f"Final test set shape: {X_test.shape}")
    print(f"All data saved to: {data_dir}")


if __name__ == "__main__":
    sys.exit(main())

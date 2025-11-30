import pandas as pd
import json
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from pathlib import Path
import sys
import os

ROOT = Path(os.getcwd())
ARTIFACTS_DIR = Path(os.getenv("ARTIFACTS_DIR", str(ROOT / "artifacts")))
VALIDATED_DIR = Path(os.getenv("VALIDATED_DIR", str(ROOT / "data" / "validated")))
PREPROCESSED_DIR = Path(os.getenv("PREPROCESSED_DIR", str(ROOT / "artifacts" / "preprocessed")))
RANDOM_STATE_VALUE = 42
TEST_SIZE = 0.2  # Simple 80/20 split for production data


def preprocess_production_data():
    """
    Preprocessing for production data retraining with replay.
    Uses simple train/test split instead of time-based splitting.
    Handles merged data (new + sampled old).
    """
    # Load data
    df = pd.read_csv(VALIDATED_DIR / "Base_validated.csv")
    print(f"Loaded merged data shape: {df.shape}")
    print(f"Class distribution:\n{df['fraud_bool'].value_counts()}")
    
    # Check if data has source marker (from merge step)
    has_source = 'data_source' in df.columns
    if has_source:
        print(f"\nData sources:\n{df['data_source'].value_counts()}")
        # Remove source marker before preprocessing
        df = df.drop(columns=['data_source'])

    # Check minimum data requirements
    min_required = 10  # Lowered for testing with small datasets
    if len(df) < min_required:
        raise ValueError(f"Insufficient data: only {len(df)} records. Need at least {min_required} for retraining.")

    # Remove month column if exists (not needed for simple split)
    if 'month' in df.columns:
        df = df.drop(columns=['month'])

    # Define features
    categorical_features = ['payment_type', 'employment_status', 'housing_status', 'source', 'device_os']
    numerical_features = [col for col in df.columns 
                         if col not in ['fraud_bool'] + categorical_features]

    # Separate features and target
    X = df.drop('fraud_bool', axis=1)
    y = df['fraud_bool']

    # Train/test split - stratified to maintain class distribution
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=TEST_SIZE, 
        random_state=RANDOM_STATE_VALUE,
        stratify=y
    )

    print(f"\nAfter split:")
    print(f"  Train size: {len(X_train)}")
    print(f"  Test size: {len(X_test)}")
    print(f"  Train class distribution:\n{y_train.value_counts()}")

    # Create preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(sparse_output=False, drop='first'), categorical_features)
        ]
    )

    # Transform features
    print("\nTransforming features...")
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    # Apply SMOTE only if we have enough samples
    min_class_count = y_train.value_counts().min()
    if min_class_count >= 6:  # SMOTE requires at least 6 samples
        print(f"\nApplying SMOTE (min class has {min_class_count} samples)...")
        smote = SMOTE(random_state=RANDOM_STATE_VALUE)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train_transformed, y_train)
        print(f"After SMOTE - Train size: {len(X_train_resampled)}")
        print(f"After SMOTE - Class distribution:\n  Class 0: {(y_train_resampled == 0).sum()}\n  Class 1: {(y_train_resampled == 1).sum()}")
    else:
        print(f"\nSkipping SMOTE (min class has only {min_class_count} samples, need at least 6)")
        X_train_resampled = X_train_transformed
        y_train_resampled = y_train

    # Generate preprocessing report
    report = generate_report(
        df_original=df,
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


def generate_report(df_original, X_train_original, y_train_original, 
                    X_train_resampled, y_train_resampled,
                    X_test_transformed, y_test,
                    categorical_features, numerical_features):
    """Generate preprocessing report."""
    
    report = {
        "data_info": {
            "original_records": int(len(df_original)),
            "original_fraud_count": int((df_original['fraud_bool'] == 1).sum()),
            "original_fraud_ratio": float((df_original['fraud_bool'] == 1).mean()),
        },
        "split_info": {
            "test_size": TEST_SIZE,
            "random_state": RANDOM_STATE_VALUE,
            "train_samples_before_smote": int(len(X_train_original)),
            "train_samples_after_smote": int(len(X_train_resampled)),
            "test_samples": int(len(X_test_transformed)),
        },
        "class_distribution": {
            "train_before_smote": {
                "class_0": int((y_train_original == 0).sum()),
                "class_1": int((y_train_original == 1).sum()),
            },
            "train_after_smote": {
                "class_0": int((y_train_resampled == 0).sum()),
                "class_1": int((y_train_resampled == 1).sum()),
            },
            "test": {
                "class_0": int((y_test == 0).sum()),
                "class_1": int((y_test == 1).sum()),
            }
        },
        "feature_info": {
            "n_features_transformed": int(X_train_resampled.shape[1]),
            "categorical_features": categorical_features,
            "numerical_features": numerical_features,
            "n_categorical": len(categorical_features),
            "n_numerical": len(numerical_features),
        },
        "preprocessing_method": "production_retraining",
        "notes": "Simplified preprocessing for production data - uses stratified train/test split"
    }
    
    return report


def save_preprocessed_data(X_train_resampled, y_train_resampled, X_test_transformed, y_test):
    """Save preprocessed data to CSV files."""
    PREPROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(X_train_resampled).to_csv(
        PREPROCESSED_DIR / "X_train_resampled.csv", index=False
    )
    pd.DataFrame(y_train_resampled, columns=["fraud_bool"]).to_csv(
        PREPROCESSED_DIR / "y_train_resampled.csv", index=False
    )
    pd.DataFrame(X_test_transformed).to_csv(
        PREPROCESSED_DIR / "X_test_transformed.csv", index=False
    )
    pd.DataFrame(y_test, columns=["fraud_bool"]).to_csv(
        PREPROCESSED_DIR / "y_test.csv", index=False
    )

    print(f"\n[INFO] Preprocessed data saved to: {PREPROCESSED_DIR}")
    print(f"  - X_train_resampled.csv: {X_train_resampled.shape}")
    print(f"  - y_train_resampled.csv: {y_train_resampled.shape}")
    print(f"  - X_test_transformed.csv: {X_test_transformed.shape}")
    print(f"  - y_test.csv: {y_test.shape}")


def save_preprocessing_report(report):
    """Save preprocessing report to JSON file."""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    
    report_path = ARTIFACTS_DIR / "preprocessing_report_production.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("[INFO] PREPROCESSING REPORT")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Report saved to: {report_path}")


def main():
    """Main execution function."""
    print("=" * 60)
    print("PRODUCTION DATA PREPROCESSING FOR RETRAINING")
    print("=" * 60)
    
    try:
        # Preprocess data
        X_train_resampled, y_train_resampled, X_test_transformed, y_test, report = preprocess_production_data()

        # Save preprocessed data
        save_preprocessed_data(X_train_resampled, y_train_resampled, X_test_transformed, y_test)

        # Save preprocessing report
        save_preprocessing_report(report)

        print("\n" + "=" * 60)
        print("PREPROCESSING COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        return 0

    except Exception as e:
        print(f"\nError during preprocessing: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

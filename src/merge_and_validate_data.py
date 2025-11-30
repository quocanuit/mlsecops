import pandas as pd
import sys
import os
from pathlib import Path
import json

ROOT = Path(os.getcwd())
RAW_DIR = Path(os.getenv("DATA_DIR_RAW", str(ROOT / "data" / "raw")))
ARTIFACTS_DIR = Path(os.getenv("ARTIFACTS_DIR", str(ROOT / "artifacts")))
VALIDATED_DIR = ARTIFACTS_DIR / "validated"

# Expected columns in correct order
EXPECTED_COLUMNS = [
    "fraud_bool", "income", "name_email_similarity", "prev_address_months_count",
    "current_address_months_count", "customer_age", "days_since_request",
    "intended_balcon_amount", "payment_type", "zip_count_4w", "velocity_6h",
    "velocity_24h", "velocity_4w", "bank_branch_count_8w",
    "date_of_birth_distinct_emails_4w", "employment_status", "credit_risk_score",
    "email_is_free", "housing_status", "phone_home_valid", "phone_mobile_valid",
    "bank_months_count", "has_other_cards", "proposed_credit_limit",
    "foreign_request", "source", "session_length_in_minutes", "device_os",
    "keep_alive_session", "device_distinct_emails_8w", "device_fraud_count", "month"
]


def load_datasets():
    """Load both new and old datasets."""
    print("=== LOADING DATASETS ===")
    
    new_data_path = RAW_DIR / "Base_new.csv"
    old_data_path = RAW_DIR / "Base_old.csv"
    
    # Load new data (required)
    if not new_data_path.exists():
        raise FileNotFoundError(f"New data not found: {new_data_path}")
    
    df_new = pd.read_csv(new_data_path)
    print(f"New production data: {df_new.shape}")
    
    # Load old data (optional)
    df_old = None
    if old_data_path.exists():
        df_old = pd.read_csv(old_data_path)
        print(f"Old training data: {df_old.shape}")
    else:
        print("Old training data not found - will use only new data")
    
    return df_new, df_old


def validate_columns(df, dataset_name="Dataset"):
    """Validate that dataset has expected columns."""
    print(f"\n=== VALIDATING {dataset_name.upper()} COLUMNS ===")
    
    df_columns = set(df.columns)
    expected_set = set(EXPECTED_COLUMNS)
    
    missing = expected_set - df_columns
    extra = df_columns - expected_set
    
    if missing:
        print(f"Warning: Missing columns in {dataset_name}: {missing}")
    
    if extra:
        print(f"Info: Extra columns in {dataset_name} (will be dropped): {extra}")
    
    # Keep only expected columns that exist
    available_columns = [col for col in EXPECTED_COLUMNS if col in df.columns]
    df = df[available_columns]
    
    print(f"{dataset_name} validated: {df.shape}")
    return df


def merge_datasets(df_new, df_old, replay_ratio=0.3):
    """Merge new data with sampled old data for replay."""
    print(f"\n=== MERGING DATASETS (Replay Ratio: {replay_ratio*100}%) ===")
    
    if df_old is None:
        print("No old data available - using only new data")
        df_merged = df_new.copy()
        df_merged['data_source'] = 'new'
        
    else:
        # Calculate sample size from old data
        n_old_samples = int(len(df_old) * replay_ratio)
        print(f"Sampling {n_old_samples} records from {len(df_old)} old records")
        
        # Sample old data (stratified by fraud_bool if possible)
        if 'fraud_bool' in df_old.columns:
            df_old_sampled = df_old.groupby('fraud_bool', group_keys=False).apply(
                lambda x: x.sample(frac=replay_ratio, random_state=42)
            )
        else:
            df_old_sampled = df_old.sample(n=n_old_samples, random_state=42)
        
        print(f"Sampled {len(df_old_sampled)} old records")
        
        # Add source markers
        df_new_marked = df_new.copy()
        df_new_marked['data_source'] = 'new'
        
        df_old_sampled_marked = df_old_sampled.copy()
        df_old_sampled_marked['data_source'] = 'old'
        
        # Merge
        df_merged = pd.concat([df_new_marked, df_old_sampled_marked], ignore_index=True)
        df_merged = df_merged.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle
        
        print(f"\nMerged dataset distribution:")
        print(df_merged['data_source'].value_counts())
    
    print(f"Final merged shape: {df_merged.shape}")
    return df_merged


def generate_validation_report(df_new, df_old, df_merged):
    """Generate validation and merge report."""
    report = {
        "new_data": {
            "records": int(len(df_new)),
            "features": int(len(df_new.columns)),
            "fraud_count": int((df_new['fraud_bool'] == 1).sum()) if 'fraud_bool' in df_new.columns else None,
            "fraud_ratio": float((df_new['fraud_bool'] == 1).mean()) if 'fraud_bool' in df_new.columns else None,
        },
        "old_data": None,
        "merged_data": {
            "total_records": int(len(df_merged)),
            "features": int(len(df_merged.columns)),
        }
    }
    
    if df_old is not None:
        report["old_data"] = {
            "original_records": int(len(df_old)),
            "features": int(len(df_old.columns)),
        }
        
        if 'data_source' in df_merged.columns:
            source_counts = df_merged['data_source'].value_counts().to_dict()
            report["merged_data"]["source_distribution"] = {
                "new": int(source_counts.get('new', 0)),
                "old": int(source_counts.get('old', 0)),
            }
    
    if 'fraud_bool' in df_merged.columns:
        report["merged_data"]["fraud_count"] = int((df_merged['fraud_bool'] == 1).sum())
        report["merged_data"]["fraud_ratio"] = float((df_merged['fraud_bool'] == 1).mean())
    
    return report


def save_merged_data(df_merged):
    """Save merged and validated data."""
    VALIDATED_DIR.mkdir(parents=True, exist_ok=True)
    
    # Remove data_source column before saving
    if 'data_source' in df_merged.columns:
        df_output = df_merged.drop(columns=['data_source'])
    else:
        df_output = df_merged
    
    output_path = VALIDATED_DIR / "Base_validated.csv"
    df_output.to_csv(output_path, index=False)
    
    print(f"\n=== SAVING MERGED DATA ===")
    print(f"Output: {output_path}")
    print(f"Shape: {df_output.shape}")
    
    return output_path


def save_report(report):
    """Save merge and validation report."""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    
    report_path = ARTIFACTS_DIR / "merge_validation_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print("\n=== MERGE & VALIDATION REPORT ===")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nReport saved to: {report_path}")


def main():
    """Main execution function."""
    print("=" * 70)
    print("MERGE AND VALIDATE DATA FOR RETRAINING")
    print("=" * 70)
    
    # Get replay ratio from environment
    replay_ratio_str = os.getenv("REPLAY_RATIO", "0.3")
    replay_ratio = float(replay_ratio_str)
    
    print(f"\nConfiguration:")
    print(f"  Replay Ratio: {replay_ratio*100}%")
    print(f"  Raw Data Dir: {RAW_DIR}")
    print(f"  Output Dir: {VALIDATED_DIR}\n")
    
    try:
        # Step 1: Load datasets
        df_new, df_old = load_datasets()
        
        # Step 2: Validate columns
        df_new = validate_columns(df_new, "New Data")
        if df_old is not None:
            df_old = validate_columns(df_old, "Old Data")
        
        # Step 3: Merge datasets with replay
        df_merged = merge_datasets(df_new, df_old, replay_ratio)
        
        # Step 4: Generate report
        report = generate_validation_report(df_new, df_old, df_merged)
        
        # Step 5: Save merged data
        save_merged_data(df_merged)
        
        # Step 6: Save report
        save_report(report)
        
        print("\n" + "=" * 70)
        print("MERGE AND VALIDATION COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        
        return 0
        
    except Exception as e:
        print(f"\nError during merge and validation: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

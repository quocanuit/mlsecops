import pandas as pd
import sys
import os
from pathlib import Path
import json

ROOT = Path(os.getcwd())
RAW_DIR = Path(os.getenv("DATA_DIR_RAW", str(ROOT / "data" / "raw")))


def load_datasets():
    print("=== LOADING DATASETS ===")

    new_data_path = RAW_DIR / "Base_new.csv"
    old_data_path = RAW_DIR / "Base.csv"

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


def merge_datasets(df_new, df_old, replay_ratio=0.3):
    print(f"\n=== MERGING DATASETS (Replay Ratio: {replay_ratio*100}%) ===")

    if df_old is None:
        print("No old data available - using only new data")
        df_merged = df_new.copy()

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

        # Add source markers for tracking
        df_new_marked = df_new.copy()
        df_new_marked['data_source'] = 'new'

        df_old_sampled_marked = df_old_sampled.copy()
        df_old_sampled_marked['data_source'] = 'old'

        # Merge
        df_merged = pd.concat([df_new_marked, df_old_sampled_marked], ignore_index=True)
        df_merged = df_merged.sample(frac=1, random_state=42).reset_index(drop=True)

        print(f"\nMerged dataset distribution:")
        print(df_merged['data_source'].value_counts())

    print(f"Final merged shape: {df_merged.shape}")
    return df_merged


def generate_merge_report(df_new, df_old, df_merged):
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
        df_temp = df_merged.drop(columns=['data_source']) if 'data_source' in df_merged.columns else df_merged
        report["merged_data"]["fraud_count"] = int((df_temp['fraud_bool'] == 1).sum())
        report["merged_data"]["fraud_ratio"] = float((df_temp['fraud_bool'] == 1).mean())

    return report


def save_merged_data(df_merged):
    output_dir = RAW_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # Remove data_source column before saving
    df_final = df_merged.drop(columns=['data_source']) if 'data_source' in df_merged.columns else df_merged

    output_path = output_dir / "Base.csv"
    df_final.to_csv(output_path, index=False)

    print(f"\n=== SAVING MERGED DATA ===")
    print(f"Output: {output_path}")
    print(f"Shape: {df_final.shape}")
    print("Note: data_source column removed before saving")

    return output_path


def save_report(report):
    """Save merge report."""
    output_dir = RAW_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    report_path = output_dir / "merge_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n=== MERGE REPORT ===")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nReport saved to: {report_path}")


def main():
    print("=" * 70)
    print("MERGE DATA FOR RETRAINING")
    print("=" * 70)

    # Get replay ratio from environment
    replay_ratio_str = os.getenv("REPLAY_RATIO", "0.3")
    replay_ratio = float(replay_ratio_str)

    print(f"\nConfiguration:")
    print(f"  Replay Ratio: {replay_ratio*100}%")
    print(f"  Raw Data Dir: {RAW_DIR}\n")

    try:
        # Step 1: Load datasets
        df_new, df_old = load_datasets()

        # Step 2: Merge datasets with replay
        df_merged = merge_datasets(df_new, df_old, replay_ratio)

        # Step 3: Generate report
        report = generate_merge_report(df_new, df_old, df_merged)

        # Step 4: Save merged data
        save_merged_data(df_merged)

        # Step 5: Save report
        save_report(report)

        print("\n" + "=" * 70)
        print("MERGE COMPLETED SUCCESSFULLY!")
        print("=" * 70)

        return 0

    except Exception as e:
        print(f"\nError during merge: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

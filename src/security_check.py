import json
import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path


def detect_label_flip(df: pd.DataFrame) -> pd.Series:
    print("Hello world!")
    return df


def main():
    ROOT = Path(os.getcwd())
    ARTIFACTS_DIR = Path(os.getenv("ARTIFACTS_DIR", str(ROOT / "artifacts")))
    DATA_DIR_RAW = Path(os.getenv("DATA_DIR_RAW", str(ROOT / "data" / "raw")))

    data_file = DATA_DIR_RAW / "Base.csv"

    print(f"Project ROOT: {ROOT}")
    if not data_file.exists():
        raise FileNotFoundError(f"CSV not found: {data_file}\nPlease put Base.csv here or update path.")

    df = pd.read_csv(data_file)

    cleaned_df = detect_label_flip(df)

    report = {
        "rows_total": int(cleaned_df.shape[0]),
    }

    output_dir = ARTIFACTS_DIR / "validated"
    os.makedirs(output_dir, exist_ok=True)

    output_file = output_dir / "Base_validated.csv"
    cleaned_df.to_csv(output_file, index=False)

    report_path = output_dir.parent / "validation_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n=== VALIDATION REPORT ===")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nSaved cleaned CSV -> {output_file}")
    print(f"Saved report      -> {report_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

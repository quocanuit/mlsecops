import json
import os
import sys
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
from pathlib import Path


EXPECTED_SCHEMA: Dict[str, str] = {
    "fraud_bool": "int64",
    "income": "float64",
    "name_email_similarity": "float64",
    "prev_address_months_count": "int64",
    "current_address_months_count": "int64",
    "customer_age": "int64",
    "days_since_request": "float64",
    "intended_balcon_amount": "float64",
    "zip_count_4w": "int64",
    "velocity_6h": "float64",
    "velocity_24h": "float64",
    "velocity_4w": "float64",
    "bank_branch_count_8w": "int64",
    "date_of_birth_distinct_emails_4w": "int64",
    "credit_risk_score": "int64",
    "email_is_free": "int64",
    "housing_status": "object",
    "phone_home_valid": "int64",
    "phone_mobile_valid": "int64",
    "bank_months_count": "int64",
    "has_other_cards": "int64",
    "proposed_credit_limit": "float64",
    "foreign_request": "int64",
    "source": "object",
    "session_length_in_minutes": "float64",
    "device_os": "object",
    "keep_alive_session": "int64",
    "device_distinct_emails_8w": "int64",
    "device_fraud_count": "int64",
    "month": "int64",
}

CONSTRAINTS = {
    "fraud_bool": {"allowed": {0, 1}},
    "customer_age": {"min": 10, "max": 90},
    "income": {"min": 0, "max": 1},
    "name_email_similarity": {"min": 0, "max": 1},
    "prev_address_months_count": {"min": -1, "max": 380},
    "current_address_months_count": {"min": -1, "max": 406},
    "days_since_request": {"min": 0},
    "intended_balcon_amount": {"min": -15, "max": 108},
    "zip_count_4w": {"min": 1},
    "velocity_6h": {"min": -211, "max": 24763},
    "velocity_24h": {"min": 1329, "max": 9527},
    "velocity_4w": {"min": 2779, "max": 7043},
    "bank_branch_count_8w": {"min": 0, "max": 2521},
    "date_of_birth_distinct_emails_4w": {"min": 0, "max": 42    },
    "credit_risk_score": {"min": -176, "max": 387},
    "email_is_free": {"allowed": {0, 1}},
    "phone_home_valid": {"allowed": {0, 1}},
    "phone_mobile_valid": {"allowed": {0, 1}},
    "bank_months_count": {"min": -1, "max": 31},
    "has_other_cards": {"allowed": {0, 1}},
    "proposed_credit_limit": {"min": 190, "max": 2000},
    "foreign_request": {"allowed": {0, 1}},
    "source": {"allowed": {"INTERNET", "APP", "TELEAPP"}},
    "keep_alive_session": {"allowed": {0, 1}},
    "session_length_in_minutes": {"min": -1, "max": 107},
    "device_os": {"allowed": {"windows", "macintosh", "linux", "x11", "other"}},
    "device_distinct_emails_8w": {"min": 0, "max": 3},
    "device_fraud_count": {"min": 0, "max": 1},
    "month": {"min": 0, "max": 7},
}


def coerce_dtypes(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    dtype_errors = []
    df = df.copy()

    for col, expected in EXPECTED_SCHEMA.items():
        if col not in df.columns:
            dtype_errors.append(f"Missing column: {col}")
            continue

        try:
            if expected == "object":
                df[col] = df[col].astype("string")

            elif expected in ("float64", "float32"):
                before_na = df[col].isna().sum()
                df[col] = pd.to_numeric(df[col], errors="coerce")
                after_na = df[col].isna().sum()
                if after_na > before_na:
                    dtype_errors.append(f"{col}: {after_na - before_na} values became NaN during float coercion")
                df[col] = df[col].astype("float64")

            elif expected in ("int64", "int32"):
                numeric = pd.to_numeric(df[col], errors="coerce")
                non_int_mask = ~numeric.dropna().apply(lambda x: float(x).is_integer())
                if non_int_mask.any():
                    dtype_errors.append(f"{col}: {non_int_mask.sum()} non-integer values found")
                df[col] = numeric.astype("Int64")

            else:
                dtype_errors.append(f"{col}: unsupported expected dtype {expected}")

        except Exception as e:
            dtype_errors.append(f"{col}: error during coercion ({e})")

    return df, dtype_errors


def validate_schema_presence(df: pd.DataFrame) -> List[str]:
    return [f"Missing column: {c}" for c in EXPECTED_SCHEMA.keys() if c not in df.columns]


def null_missing_report(df: pd.DataFrame) -> pd.Series:
    return df.isna().sum().sort_values(ascending=False)


def build_valid_mask(df: pd.DataFrame) -> pd.Series:
    mask = pd.Series(True, index=df.index)
    existing_cols = [c for c in EXPECTED_SCHEMA.keys() if c in df.columns]
    missing_cols = [c for c in EXPECTED_SCHEMA.keys() if c not in df.columns]

    if missing_cols:
        print(f"[WARN] Missing columns skipped in NA check: {missing_cols}")

    if existing_cols:
        mask &= df[existing_cols].notna().all(axis=1)

    for col, rule in CONSTRAINTS.items():
        if col not in df.columns:
            continue
        col_vals = df[col]
        if "allowed" in rule:
            mask &= col_vals.isin(list(rule["allowed"]))
        if "min" in rule:
            mask &= col_vals >= rule["min"]
        if "max" in rule:
            mask &= col_vals <= rule["max"]

    return mask


def main():
    ROOT = Path(__file__).resolve().parent
    data_file = ROOT / "data" / "raw" / "Base.csv"

    print(f"Project ROOT: {ROOT}")
    if not data_file.exists():
        raise FileNotFoundError(f"CSV not found: {data_file}\n→ Please put Base.csv here or update path.")

    df = pd.read_csv(data_file)

    schema_presence_errors = validate_schema_presence(df)
    df, dtype_errors = coerce_dtypes(df)
    missing = null_missing_report(df)
    valid_mask = build_valid_mask(df)

    cleaned = df[valid_mask].copy()

    fraud_distribution = {}
    if "fraud_bool" in cleaned.columns:
        fraud_distribution = {str(k): int(v) for k, v in cleaned["fraud_bool"].value_counts(dropna=False).items()}

    report = {
        "rows_total": int(df.shape[0]),
        "schema_missing_columns": schema_presence_errors,
        "dtype_errors_summary": dtype_errors,
        "missing_values_summary": {str(k): int(v) for k, v in missing[missing > 0].items()},
        "invalid_rows_count": int((~valid_mask).sum()),
        "fraud_bool_distribution": fraud_distribution,
    }

    for col, typ in EXPECTED_SCHEMA.items():
        if col not in cleaned.columns:
            continue
        try:
            if typ in ("int64", "int32"):
                cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce").astype("Int64")
            elif typ == "object":
                cleaned[col] = cleaned[col].astype("string").astype("object")
        except Exception as e:
            print(f"[WARN] Column {col} casting skipped ({e})")

    output_dir = ROOT / "artifacts" / "validated_data"
    os.makedirs(output_dir, exist_ok=True)

    output_file = output_dir / "train_processed.csv"
    cleaned.to_csv(output_file, index=False)

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

import json
import os
import sys
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
from pathlib import Path


EXPECTED_SCHEMA: Dict[str, str] = {
    "case_id": "int64",
    "fraud_bool": "int64",
    "income": "float64",
    "name_email_similarity": "float64",
    "prev_address_months_count": "int64",
    "current_address_months_count": "int64",
    "customer_age": "int64",
    "days_since_request": "float64",
    "intended_balcon_amount": "float64",
    "payment_type": "object",
    "zip_count_4w": "int64",
    "velocity_6h": "float64",
    "velocity_24h": "float64",
    "velocity_4w": "float64",
    "bank_branch_count_8w": "int64",
    "date_of_birth_distinct_emails_4w": "int64",
    "employment_status": "object",
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
    "model_score": "float64",
    "batch": "int64",
    "assignment": "object",
    "decision": "float64",
}

CONSTRAINTS = {
    "fraud_bool": {"allowed": {0, 1}},
    "customer_age": {"min": 15, "max": 120},
    "income": {"min": 0, "max": 1},
    "name_email_similarity": {"min": 0, "max": 1},
    "prev_address_months_count": {"min": -1},
    "current_address_months_count": {"min": -1},
    "days_since_request": {"min": 0},
    "intended_balcon_amount": {"min": -1},
    "payment_type": {"allowed": {"AA", "AB", "AC", "AD"}},
    "zip_count_4w": {"min": 1},
    "velocity_6h": {"min": 0},
    "velocity_24h": {"min": 0},
    "velocity_4w": {"min": 0},
    "bank_branch_count_8w": {"min": 0},
    "date_of_birth_distinct_emails_4w": {"min": 0},
    "employment_status": {"allowed": {"CA", "CB", "CC", "CD", "CE", "CF"}},
    "credit_risk_score": {"min": -176},
    "email_is_free": {"allowed": {0, 1}},
    "phone_home_valid": {"allowed": {0, 1}},
    "phone_mobile_valid": {"allowed": {0, 1}},
    "bank_months_count": {"min": -1},
    "has_other_cards": {"allowed": {0, 1}},
    "proposed_credit_limit": {"min": 0},
    "foreign_request": {"allowed": {0, 1}},
    "source": {"allowed": {"INTERNET", "APP"}},
    "keep_alive_session": {"allowed": {0, 1}},
    "session_length_in_minutes": {"min": -1},
    "device_os": {"allowed": {"windows", "acintox", "linux", "x11", "other"}},
    "device_distinct_emails_8w": {"min": 0},
    "device_fraud_count": {"min": 0, "max": 1},
    "month": {"min": 1, "max": 12},
    "decision": {"min": 0.0, "max": 1.0},
}


def coerce_dtypes(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    dtype_errors = []
    df = df.copy()

    for col, expected in EXPECTED_SCHEMA.items():
        if col not in df.columns:
            dtype_errors.append(f"Missing column: {col}")
            continue

        if expected == "object":
            try:
                df[col] = df[col].astype("string")
            except Exception as e:
                dtype_errors.append(f"{col}: cannot cast to string ({e})")

        elif expected in ("float64", "float32"):
            before_na = df[col].isna().sum()
            df[col] = pd.to_numeric(df[col], errors="coerce")
            after_na = df[col].isna().sum()
            if after_na > before_na:
                dtype_errors.append(
                    f"{col}: {after_na - before_na} values became NaN during float coercion"
                )
            df[col] = df[col].astype("float64")

        elif expected in ("int64", "int32"):
            numeric = pd.to_numeric(df[col], errors="coerce")
            non_int_mask = ~numeric.dropna().apply(lambda x: float(x).is_integer())
            if non_int_mask.any():
                dtype_errors.append(
                    f"{col}: {non_int_mask.sum()} non-integer values found"
                )
            df[col] = numeric.astype("Int64")

        else:
            dtype_errors.append(f"{col}: unsupported expected dtype {expected}")

    return df, dtype_errors


def validate_schema_presence(df: pd.DataFrame) -> List[str]:
    errors = []
    for col in EXPECTED_SCHEMA.keys():
        if col not in df.columns:
            errors.append(f"Missing column: {col}")
    return errors


def null_missing_report(df: pd.DataFrame) -> pd.Series:
    return df.isna().sum().sort_values(ascending=False)


def build_valid_mask(df: pd.DataFrame) -> pd.Series:
    mask = pd.Series(True, index=df.index)

    required_cols = list(EXPECTED_SCHEMA.keys())
    mask &= df[required_cols].notna().all(axis=1)

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

    ROOT = Path(os.getcwd())
    print(f"Project ROOT: {ROOT}")
    df = pd.read_csv(f"{ROOT}/ICAIF_KAGGLE/raw/train.csv")

    schema_presence_errors = validate_schema_presence(df)
    df, dtype_errors = coerce_dtypes(df)
    missing = null_missing_report(df)
    valid_mask = build_valid_mask(df)

    report = {
        "rows_total": int(df.shape[0]),
        "schema_missing_columns": schema_presence_errors,
        "dtype_errors_summary": dtype_errors,
        "missing_values_summary": missing[missing > 0].to_dict(),
        "invalid_rows_count": int((~valid_mask).sum()),
    }

    cleaned = df[valid_mask].copy()

    for col, typ in EXPECTED_SCHEMA.items():
        if col in cleaned.columns and typ in ("int64", "int32"):
            try:
                cleaned[col] = cleaned[col].astype("Int64").astype("int64")
            except Exception:
                pass
        if col in cleaned.columns and typ == "object":
            try:
                cleaned[col] = cleaned[col].astype("string").astype("object")
            except Exception:
                pass


    output_dir = ROOT / "validated_data"
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    output_file = output_dir / "train_processed.csv"
    cleaned.to_csv(output_file, index=False)

    report_path = output_dir / "validation_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("=== VALIDATION REPORT ===")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nSaved cleaned CSV -> {output_file}")
    print(f"Saved report      -> {report_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
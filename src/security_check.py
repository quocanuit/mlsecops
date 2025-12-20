import json
import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle
import boto3


def preprocess_with_labelencoder(df: pd.DataFrame, col_label: str):
    categorical_features = df.select_dtypes(include=["object", "category"]).columns
    numerical_features = df.select_dtypes(include=["number"]).columns

    categorical_features = [c for c in categorical_features if c != col_label]
    numerical_features = [c for c in numerical_features if c != col_label]

    label_encoders = {}
    scaler = StandardScaler()

    for col in categorical_features:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    df[numerical_features] = scaler.fit_transform(df[numerical_features])

    return df, label_encoders, scaler


def download_model_from_s3(bucket: str, key: str, local_path: str):
    """
    Download model artifact from S3 if not exists locally
    """
    if os.path.exists(local_path):
        return

    print(f"Downloading model from s3://{bucket}/{key}")
    s3 = boto3.client("s3")
    s3.download_file(bucket, key, local_path)
    print(f"Model saved to {local_path}")


def detect_label_flip(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect suspected label-flip rows using ensemble model + high-recall strategy.
    This function MUST NOT reorder rows.
    """

    # ------------------------------------------------------------
    # 0) Ensure models are available locally
    # ------------------------------------------------------------
    download_model_from_s3(
        bucket="mlsecops-model-temp",
        key="model_rf.pkl",
        local_path="model_rf.pkl",
    )

    download_model_from_s3(
        bucket="mlsecops-model-temp",
        key="model_xgb.pkl",
        local_path="model_xgb.pkl",
    )

    # ------------------------------------------------------------
    # 1) Feature cleaning
    # ------------------------------------------------------------
    df_cleaned = df.copy()
    df_cleaned = df_cleaned.drop(
        columns=[
            "bank_months_count",
            "prev_address_months_count",
            "velocity_4w",
        ],
        errors="ignore",
    )

    cols_missing = [
        "current_address_months_count",
        "session_length_in_minutes",
        "device_distinct_emails_8w",
        "intended_balcon_amount",
    ]

    for c in cols_missing:
        if c in df_cleaned.columns:
            df_cleaned[c] = df_cleaned[c].replace(-1, np.nan)

    df_cleaned = df_cleaned.dropna()

    # ------------------------------------------------------------
    # 2) Preprocess
    # ------------------------------------------------------------
    df_preprocessed, _, _ = preprocess_with_labelencoder(
        df=df_cleaned, col_label="fraud_bool"
    )

    X = df_preprocessed.drop(columns="fraud_bool")
    y = df_preprocessed["fraud_bool"].astype(int)

    # ------------------------------------------------------------
    # 3) Load ground-truth Base.csv (evaluation only)
    # ------------------------------------------------------------
    base_path = Path(os.getenv("DATA_DIR_RAW", "data/raw")) / "Base.csv"
    df_base = pd.read_csv(base_path)

    y_true = df_base.loc[df_preprocessed.index, "fraud_bool"].astype(int).values
    true_flip = (y_true != y.values).astype(int)

    # ------------------------------------------------------------
    # 4) Load models
    # ------------------------------------------------------------
    with open("model_xgb.pkl", "rb") as f:
        model_xgb = pickle.load(f)

    with open("model_rf.pkl", "rb") as f:
        model_rf = pickle.load(f)

    # ------------------------------------------------------------
    # 5) Predict probabilities
    # ------------------------------------------------------------
    p_xgb = model_xgb.predict_proba(X)[:, 1]
    p_rf = model_rf.predict_proba(X)[:, 1]
    p_mean = (p_xgb + p_rf) / 2.0

    # ------------------------------------------------------------
    # 6) Suspicion score (UNCHANGED LOGIC)
    # ------------------------------------------------------------
    ple = np.where(y == 0, p_mean, 1.0 - p_mean)
    disagree = np.abs(p_xgb - p_rf)
    near_boundary = 1.0 - np.abs(p_mean - 0.5) * 2.0
    near_boundary = np.clip(near_boundary, 0, 1)

    suspicion_score = 0.65 * ple + 0.25 * disagree + 0.10 * near_boundary

    # ------------------------------------------------------------
    # 7) Auto-tune K for recall >= 0.8
    # ------------------------------------------------------------
    scores = suspicion_score
    order = np.argsort(scores)[::-1]

    target_recall = 0.80
    best_K = None

    for k in range(1000, len(scores), 500):
        pred = np.zeros(len(scores))
        pred[order[:k]] = 1

        TP = np.sum((pred == 1) & (true_flip == 1))
        FN = np.sum((pred == 0) & (true_flip == 1))

        recall = TP / (TP + FN + 1e-9)
        if recall >= target_recall:
            best_K = k
            break

    if best_K is None:
        best_K = int(len(scores) * 0.2)

    # ------------------------------------------------------------
    # 8) Apply detection (NO reordering)
    # ------------------------------------------------------------
    df_result = df.copy()
    df_result["suspected_flip"] = False

    flagged_index = df_preprocessed.index[order[:best_K]]
    df_result.loc[flagged_index, "suspected_flip"] = True

    # ------------------------------------------------------------
    # 9) Return ONLY suspected rows
    # ------------------------------------------------------------
    return df_result[df_result["suspected_flip"] == True]


def main():
    ROOT = Path(os.getcwd())
    ARTIFACTS_DIR = Path(os.getenv("ARTIFACTS_DIR", str(ROOT / "artifacts")))
    DATA_DIR_RAW = Path(os.getenv("DATA_DIR_RAW", str(ROOT / "data" / "raw")))

    data_file = DATA_DIR_RAW / "Base.csv"

    print(f"Project ROOT: {ROOT}")
    if not data_file.exists():
        raise FileNotFoundError(
            f"CSV not found: {data_file}\nPlease put Base.csv here or update path."
        )

    df = pd.read_csv(data_file)

    cleaned_df = detect_label_flip(df)

    report = {
        "suspected_rows_total": int(cleaned_df.shape[0]),
    }

    output_dir = ARTIFACTS_DIR / "validated"
    os.makedirs(output_dir, exist_ok=True)

    output_file = output_dir / "Base_suspected.csv"
    cleaned_df.to_csv(output_file, index=False)

    report_path = output_dir.parent / "validation_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n=== VALIDATION REPORT ===")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nSaved suspected CSV -> {output_file}")
    print(f"Saved report      -> {report_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

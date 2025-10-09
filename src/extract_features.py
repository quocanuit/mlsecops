import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.ensemble import IsolationForest
from pathlib import Path
import os

ROOT = Path(os.getcwd())
print(f"Project ROOT: {ROOT}")
df = pd.read_csv(f"{ROOT}/ICAIF_KAGGLE/testbed/train/small__regular/train.csv")

# Ensure categorical types
cat_cols_raw = ["payment_type","employment_status","housing_status","source","device_os","assignment"]
for c in cat_cols_raw:
    if c in df.columns:
        df[c] = df[c].astype("string")

eps = 1e-6

# Ratios from velocities
if {"velocity_6h","velocity_24h","velocity_4w"}.issubset(df.columns):
    df["v6h_per_24h"] = df["velocity_6h"] / (df["velocity_24h"] + eps)
    df["v24h_per_4w"] = df["velocity_24h"] / (df["velocity_4w"] + eps)
    df["v6h_per_4w"]  = df["velocity_6h"] / (df["velocity_4w"] + eps)

# Address stability proxy
if {"current_address_months_count","prev_address_months_count"}.issubset(df.columns):
    df["address_stability"] = (df["current_address_months_count"].clip(lower=0) + 1) / \
                               (df["prev_address_months_count"].clip(lower=0) + 1)

# Phone validity union
if {"phone_home_valid","phone_mobile_valid"}.issubset(df.columns):
    df["any_phone_valid"] = ((df["phone_home_valid"]==1) | (df["phone_mobile_valid"]==1)).astype(int)

# Device seen fraud
if "device_fraud_count" in df.columns:
    df["device_seen_fraud"] = (df["device_fraud_count"] > 0).astype(int)

# Exposure ratio
if {"proposed_credit_limit","income"}.issubset(df.columns):
    df["limit_to_income"] = df["proposed_credit_limit"] / (df["income"] + eps)

# Simple interaction
if {"email_is_free","foreign_request"}.issubset(df.columns):
    df["free_email_x_foreign"] = df["email_is_free"] * df["foreign_request"]

# Log1p transforms for heavy-tailed numerics
for c in ["zip_count_4w","velocity_6h","velocity_24h","velocity_4w",
          "bank_branch_count_8w","device_distinct_emails_8w",
          "proposed_credit_limit","intended_balcon_amount","session_length_in_minutes"]:
    if c in df.columns:
        df[c+"_log1p"] = np.log1p(np.clip(df[c], a_min=0, a_max=None))

# Z-scores for some numeric fields
for c in ["model_score","decision","credit_risk_score","customer_age","bank_months_count"]:
    if c in df.columns:
        col = df[c].astype(float)
        mu, sigma = col.mean(), col.std() + 1e-6
        df[c+"_z"] = (col - mu) / sigma

# Outlier & anomaly signals
num_only = df.select_dtypes(exclude=["object","string"]).fillna(0)
iso = IsolationForest(random_state=42, contamination=0.01, n_jobs=-1)
df["iforest_score"] = -iso.fit_predict(num_only)

# PCA latent components
scaler = StandardScaler()
num_scaled = scaler.fit_transform(num_only)
pca = PCA(n_components=10, random_state=42)
pca_feats = pca.fit_transform(num_scaled)
for i in range(pca_feats.shape[1]):
    df[f"pca_component_{i+1}"] = pca_feats[:, i]

# Drop weak bases
to_drop_bases = [
    "keep_alive_session",
    "proposed_credit_limit",
    "proposed_credit_limit_log1p",
    "session_length_in_minutes",
    "income",
    "customer_age",
    "credit_risk_score_z",
    "limit_to_income",
    "velocity_24h",
    "address_stability",
    "phone_home_valid",
    "device_os",
    "housing_status",
    "employment_status",
]
X = df.drop(columns=[c for c in ["fraud_bool","case_id"]+to_drop_bases if c in df.columns], errors="ignore")
y = df["fraud_bool"].astype(int)

# Identify categorical vs numeric
cat_cols = [c for c in ["payment_type","source","assignment"] if c in X.columns]
num_cols = X.columns.difference(cat_cols).tolist()

# Preprocess pipeline
num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median"))])
cat_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])
preprocess = ColumnTransformer(
    transformers=[
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols),
    ],
    remainder="drop"
)

# --- Split stratified ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

print("Train:", X_train.shape, " Test:", X_test.shape)

extracted_data_dir = ROOT / "extracted_data"
extracted_data_dir.mkdir(exist_ok=True)

# Save training and test sets
X_train.to_csv(extracted_data_dir / "X_train.csv", index=False)
X_test.to_csv(extracted_data_dir / "X_test.csv", index=False)
y_train.to_csv(extracted_data_dir / "y_train.csv", index=False)
y_test.to_csv(extracted_data_dir / "y_test.csv", index=False)

print(f"Saved extracted data to: {extracted_data_dir}")
print(f"Files created: X_train.csv, X_test.csv, y_train.csv, y_test.csv")

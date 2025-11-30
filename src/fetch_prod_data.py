import sys
import os
import boto3
from pathlib import Path
import pandas as pd

# Column mapping: from DynamoDB to expected format
COLUMN_ORDER = [
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


def fetch_from_dynamodb(table_name: str, region_name: str = "us-east-1", max_items: int = None):
    print(f"Connecting to DynamoDB table: {table_name}")

    dynamodb = boto3.resource('dynamodb', region_name=region_name)
    table = dynamodb.Table(table_name)

    print(f"Scanning table (max items: {max_items if max_items else 'unlimited'})...")

    scan_kwargs = {}
    if max_items:
        scan_kwargs['Limit'] = max_items

    response = table.scan(**scan_kwargs)
    items = response['Items']

    # Handle pagination if no limit or need more items
    while 'LastEvaluatedKey' in response and (not max_items or len(items) < max_items):
        print(f"Fetched {len(items)} items so far...")

        if max_items:
            remaining = max_items - len(items)
            scan_kwargs['Limit'] = min(remaining, 1000)  # DynamoDB max page size

        scan_kwargs['ExclusiveStartKey'] = response['LastEvaluatedKey']
        response = table.scan(**scan_kwargs)
        items.extend(response['Items'])

        if max_items and len(items) >= max_items:
            items = items[:max_items]
            break

    print(f"Total items fetched: {len(items)}")
    return items


def transform_to_dataframe(items):
    print("Transforming data to DataFrame...")

    # Convert to DataFrame
    df = pd.DataFrame(items)

    print(f"Original shape: {df.shape}")
    print(f"Original columns: {list(df.columns)}")

    # Select and reorder columns
    available_columns = [col for col in COLUMN_ORDER if col in df.columns]
    missing_columns = [col for col in COLUMN_ORDER if col not in df.columns]

    if missing_columns:
        print(f"Warning: Missing columns: {missing_columns}")

    df = df[available_columns]

    print(f"Final shape: {df.shape}")
    print(f"Final columns: {list(df.columns)}")

    return df


def save_to_local(df, local_path: str):
    print(f"Saving data to: {local_path}")

    os.makedirs(os.path.dirname(local_path), exist_ok=True)

    # Save to CSV
    df.to_csv(local_path, index=False)

    print(f"Successfully saved to {local_path}")


def main():
    print("=== FETCHING NEW PRODUCTION DATA FROM DYNAMODB ===\n")

    # Configuration from environment variables
    table_name = os.getenv("DYNAMODB_TABLE", "silver-table")
    region = os.getenv("AWS_REGION", "us-east-1")
    output_path = os.getenv("OUTPUT_PATH", "/workspace/data/raw/Base_new.csv")

    # Max items to fetch (for controlling data volume)
    max_items_str = os.getenv("MAX_ITEMS", "")
    max_items = int(max_items_str) if max_items_str else None

    print(f"Configuration:")
    print(f"  DynamoDB Table: {table_name}")
    print(f"  Output Path: {output_path}")
    print(f"  Max Items: {max_items if max_items else 'unlimited'}")
    print(f"  Region: {region}\n")

    try:
        # Step 1: Fetch data from DynamoDB
        items = fetch_from_dynamodb(table_name, region, max_items)

        if not items:
            print("Error: No data found in DynamoDB table")
            return 1

        # Step 2: Transform to DataFrame
        df = transform_to_dataframe(items)

        # Step 3: Save to local file
        save_to_local(df, output_path)

        # Print summary
        print("\n=== SUMMARY ===")
        print(f"Records processed: {len(df)}")
        print(f"Features: {len(df.columns)}")
        print(f"Output location: {output_path}")
        print("\nNew data fetch completed successfully!")

        return 0

    except Exception as e:
        print(f"\nError during execution: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

import sys
import os
import boto3
from pathlib import Path
import pandas as pd
from datetime import datetime

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


def fetch_from_dynamodb(table_name: str, region_name: str = "us-east-1"):
    """Fetch all data from DynamoDB table."""
    print(f"Connecting to DynamoDB table: {table_name}")
    
    dynamodb = boto3.resource('dynamodb', region_name=region_name)
    table = dynamodb.Table(table_name)
    
    print("Scanning table...")
    response = table.scan()
    items = response['Items']
    
    # Handle pagination
    while 'LastEvaluatedKey' in response:
        print(f"Fetched {len(items)} items so far...")
        response = table.scan(ExclusiveStartKey=response['LastEvaluatedKey'])
        items.extend(response['Items'])
    
    print(f"Total items fetched: {len(items)}")
    return items


def transform_to_dataframe(items):
    """Transform DynamoDB items to pandas DataFrame with correct column order."""
    print("Transforming data to DataFrame...")
    
    # Convert to DataFrame
    df = pd.DataFrame(items)
    
    print(f"Original shape: {df.shape}")
    print(f"Original columns: {list(df.columns)}")
    
    # Select and reorder columns
    # Only keep columns that exist in both the data and the expected order
    available_columns = [col for col in COLUMN_ORDER if col in df.columns]
    missing_columns = [col for col in COLUMN_ORDER if col not in df.columns]
    
    if missing_columns:
        print(f"Warning: Missing columns: {missing_columns}")
    
    df = df[available_columns]
    
    print(f"Final shape: {df.shape}")
    print(f"Final columns: {list(df.columns)}")
    
    return df


def save_to_s3(df, s3_bucket: str, s3_key: str):
    """Save DataFrame to S3 as CSV."""
    print(f"Saving data to S3: s3://{s3_bucket}/{s3_key}")
    
    # Create temporary file
    temp_file = "/tmp/production_data.csv"
    df.to_csv(temp_file, index=False)
    
    # Upload to S3
    s3_client = boto3.client('s3')
    s3_client.upload_file(temp_file, s3_bucket, s3_key)
    
    print(f"Successfully uploaded to S3")
    
    # Clean up
    os.remove(temp_file)


def main():
    """Main execution function."""
    print("=== FETCHING PRODUCTION DATA FROM DYNAMODB ===\n")
    
    # Configuration from environment variables
    table_name = os.getenv("DYNAMODB_TABLE", "silver-table")
    s3_bucket = os.getenv("S3_BUCKET", "mlsecops-production-data")
    region = os.getenv("AWS_REGION", "us-east-1")
    
    # Generate timestamped filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    s3_key = f"gold/production_data_{timestamp}.csv"
    
    print(f"Configuration:")
    print(f"  DynamoDB Table: {table_name}")
    print(f"  S3 Bucket: {s3_bucket}")
    print(f"  S3 Key: {s3_key}")
    print(f"  Region: {region}\n")
    
    try:
        # Step 1: Fetch data from DynamoDB
        items = fetch_from_dynamodb(table_name, region)
        
        if not items:
            print("Error: No data found in DynamoDB table")
            return 1
        
        # Step 2: Transform to DataFrame
        df = transform_to_dataframe(items)
        
        # Step 3: Save to S3
        save_to_s3(df, s3_bucket, s3_key)
        
        # Print summary
        print("\n=== SUMMARY ===")
        print(f"Records processed: {len(df)}")
        print(f"Features: {len(df.columns)}")
        print(f"Output location: s3://{s3_bucket}/{s3_key}")
        print("\nData fetch completed successfully!")
        
        return 0
        
    except Exception as e:
        print(f"\nError during execution: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

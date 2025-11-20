#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path
import boto3


def download_file(bucket, workflow_name, s3_key, local_path):
    Path(local_path).parent.mkdir(parents=True, exist_ok=True)
    full_key = f"{workflow_name}/{s3_key}"
    print(f"Downloading s3://{bucket}/{full_key}")
    boto3.client('s3').download_file(bucket, full_key, local_path)
    print(f"Downloaded to {local_path}")


def upload_file(bucket, workflow_name, local_path, s3_key):
    full_key = f"{workflow_name}/{s3_key}"
    print(f"Uploading {local_path} to s3://{bucket}/{full_key}")
    boto3.client('s3').upload_file(local_path, bucket, full_key)
    print(f"Uploaded")


def download_dir(bucket, workflow_name, s3_prefix, local_dir, pattern="*.csv"):
    local_path = Path(local_dir)
    local_path.mkdir(parents=True, exist_ok=True)

    full_prefix = f"{workflow_name}/{s3_prefix}/"
    print(f"Downloading from s3://{bucket}/{full_prefix}")

    s3 = boto3.client('s3')
    paginator = s3.get_paginator('list_objects_v2')

    count = 0
    for page in paginator.paginate(Bucket=bucket, Prefix=full_prefix):
        for obj in page.get('Contents', []):
            key = obj['Key']
            rel_path = key[len(full_prefix):]
            if not rel_path or (pattern != "*" and not rel_path.endswith(pattern.replace("*.", "."))):
                continue

            dest = local_path / rel_path
            dest.parent.mkdir(parents=True, exist_ok=True)
            s3.download_file(bucket, key, str(dest))
            print(f"  {rel_path}")
            count += 1

    print(f"Downloaded {count} file(s)")


def upload_dir(bucket, workflow_name, local_dir, s3_prefix, pattern="*.csv"):
    local_path = Path(local_dir)
    full_prefix = f"{workflow_name}/{s3_prefix}/"
    print(f"Uploading to s3://{bucket}/{full_prefix}")

    s3 = boto3.client('s3')
    count = 0

    for file in local_path.rglob(pattern.replace("*.", "*.")):
        if file.is_file():
            rel_path = file.relative_to(local_path)
            s3_key = full_prefix + str(rel_path)
            s3.upload_file(str(file), bucket, s3_key)
            print(f"  {rel_path}")
            count += 1

    print(f"Uploaded {count} file(s)")

def main():
    parser = argparse.ArgumentParser(description="S3 artifacts handler")
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--workflow-name", required=True)

    subparsers = parser.add_subparsers(dest="command")

    dl_file = subparsers.add_parser("download-file")
    dl_file.add_argument("--s3-key", required=True)
    dl_file.add_argument("--local-path", required=True)

    up_file = subparsers.add_parser("upload-file")
    up_file.add_argument("--local-path", required=True)
    up_file.add_argument("--s3-key", required=True)

    dl_dir = subparsers.add_parser("download-dir")
    dl_dir.add_argument("--s3-prefix", required=True)
    dl_dir.add_argument("--local-dir", required=True)
    dl_dir.add_argument("--pattern", default="*.csv")

    up_dir = subparsers.add_parser("upload-dir")
    up_dir.add_argument("--local-dir", required=True)
    up_dir.add_argument("--s3-prefix", required=True)
    up_dir.add_argument("--pattern", default="*.csv")

    args = parser.parse_args()

    if args.command == "download-file":
        download_file(args.bucket, args.workflow_name, args.s3_key, args.local_path)
    elif args.command == "upload-file":
        upload_file(args.bucket, args.workflow_name, args.local_path, args.s3_key)
    elif args.command == "download-dir":
        download_dir(args.bucket, args.workflow_name, args.s3_prefix, args.local_dir, args.pattern)
    elif args.command == "upload-dir":
        upload_dir(args.bucket, args.workflow_name, args.local_dir, args.s3_prefix, args.pattern)


if __name__ == "__main__":
    main()

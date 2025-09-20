#!/bin/bash

set -e

if [ -z "$AWS_ACCESS_KEY_ID" ] || [ -z "$AWS_SECRET_ACCESS_KEY" ] || [ -z "$AWS_REGION" ]; then
    echo "Error: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, and AWS_REGION not set." >&2
    exit 1
fi

pip install -r "requirements.txt"

aws configure set aws_access_key_id "$AWS_ACCESS_KEY_ID" --profile mlsecops_user
aws configure set aws_secret_access_key "$AWS_SECRET_ACCESS_KEY" --profile mlsecops_user
aws configure set region "$AWS_REGION" --profile mlsecops_user
aws configure set output "text" --profile mlsecops_user
echo "AWS CLI profile 'mlsecops_user' configured."

export AWS_PROFILE=mlsecops_user

dvc pull

DATASET_PATH="$(pwd)/ICAIF_KAGGLE"
echo "Dataset path: $DATASET_PATH"

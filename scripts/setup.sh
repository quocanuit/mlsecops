#!/bin/bash

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CLUSTER_NAME="mlsecops-eks-cluster"
REGION="us-east-1"
NAMESPACE="argocd"

echo "ArgoCD EKS Access Setup"
echo ""

echo "Checking required tools..."
command -v aws >/dev/null 2>&1 || { echo "AWS CLI is required but not installed. Aborting."; return 1 2>/dev/null || exit 1; }
command -v kubectl >/dev/null 2>&1 || { echo "kubectl is required but not installed. Aborting."; return 1 2>/dev/null || exit 1; }
command -v jq >/dev/null 2>&1 || { echo "jq is required but not installed. Aborting."; return 1 2>/dev/null || exit 1; }
echo "All required tools are installed"

# Get caller identity
CALLER_IDENTITY=$(aws sts get-caller-identity)
ACCOUNT_ID=$(echo $CALLER_IDENTITY | jq -r '.Account')
CALLER_ARN=$(echo $CALLER_IDENTITY | jq -r '.Arn')

# Check if already using the target role
ROLE_ARN="arn:aws:iam::${ACCOUNT_ID}:role/GithubActions"

if [[ "$CALLER_ARN" == *":assumed-role/GithubActions/"* ]]; then
    echo "Already using required role, skipping role assumption"
else
    # Assume role
    SESSION_NAME="dev-session-$(date +%s)"

    echo "Assuming role: $ROLE_ARN"
    echo "Session name: $SESSION_NAME"

    CREDENTIALS=$(aws sts assume-role \
        --role-arn $ROLE_ARN \
        --role-session-name $SESSION_NAME \
        --output json)

    if [ $? -ne 0 ]; then
        echo "Failed to assume role. Please check your permissions."
        return 1 2>/dev/null || exit 1
    fi

    echo "Role assumed successfully"

    export AWS_ACCESS_KEY_ID=$(echo $CREDENTIALS | jq -r '.Credentials.AccessKeyId')
    export AWS_SECRET_ACCESS_KEY=$(echo $CREDENTIALS | jq -r '.Credentials.SecretAccessKey')
    export AWS_SESSION_TOKEN=$(echo $CREDENTIALS | jq -r '.Credentials.SessionToken')
fi

# Update kubeconfig
echo "Updating kubeconfig for EKS cluster: $CLUSTER_NAME"
aws eks update-kubeconfig \
    --name $CLUSTER_NAME \
    --region $REGION

if [ $? -ne 0 ]; then
    echo "Failed to update kubeconfig. Please check if the cluster exists."
    return 1 2>/dev/null || exit 1
fi

echo "Kubeconfig updated"

# Check cluster access
echo "Verifying cluster access..."
kubectl get nodes

if [ $? -ne 0 ]; then
    echo "Failed to access cluster. Please check your permissions."
    return 1 2>/dev/null || exit 1
fi

echo "Successfully connected to EKS cluster"

echo ""
echo "To access services, see: docs/service-access.md"

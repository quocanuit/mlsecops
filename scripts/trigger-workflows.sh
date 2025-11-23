#!/bin/bash

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "Argo Workflows Trigger Script"
echo ""

echo "Checking required tools..."
command -v argo >/dev/null 2>&1 || { echo "argo CLI is required but not installed. Aborting."; return 1 2>/dev/null || exit 1; }
command -v kubectl >/dev/null 2>&1 || { echo "kubectl is required but not installed. Aborting."; return 1 2>/dev/null || exit 1; }
command -v yq >/dev/null 2>&1 || { echo "yq is required but not installed. Aborting."; return 1 2>/dev/null || exit 1; }
echo "All required tools are installed"

# Verify cluster access
echo "Verifying cluster access..."
kubectl get nodes

if [ $? -ne 0 ]; then
    echo "Failed to access cluster. Please check your permissions."
    return 1 2>/dev/null || exit 1
fi

echo "Successfully connected to cluster"

# Load configuration
CONFIG_FILE="$ROOT/scripts/cd-config.yaml"
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Configuration file $CONFIG_FILE not found."
    return 1 2>/dev/null || exit 1
fi

# Parse arguments
case "$1" in
    --apply-templates)
        echo "Applying workflow templates..."
        kubectl apply -f "$ROOT/tools/workflows/templates"
        if [ $? -ne 0 ]; then
            echo "Failed to apply templates."
            return 1 2>/dev/null || exit 1
        fi
        echo "Templates applied successfully."
        ;;
    --training-pipeline)
        echo "Submitting training pipeline..."
        IMAGE_TAG=$(yq e '.trainingPipeline.imageTag' "$CONFIG_FILE")
        ECR_REGISTRY=$(yq e '.trainingPipeline.ecrRegistry' "$CONFIG_FILE")
        MLFLOW_TRACKING_URI=$(yq e '.trainingPipeline.mlflowTrackingUrl' "$CONFIG_FILE")
        S3_ARTIFACT_BUCKET=$(yq e '.trainingPipeline.s3ArtifactBucket' "$CONFIG_FILE")

        argo submit "$ROOT/tools/workflows/training-pipeline.yaml" \
            -n argo-workflows \
            -p image-tag="$IMAGE_TAG" \
            -p ecr-registry="$ECR_REGISTRY" \
            -p mlflow-tracking-uri="$MLFLOW_TRACKING_URI" \
            -p s3-artifact-bucket="$S3_ARTIFACT_BUCKET"
        if [ $? -ne 0 ]; then
            echo "Failed to submit training pipeline."
            return 1 2>/dev/null || exit 1
        fi
        echo "Training pipeline submitted successfully."
        ;;
    --serving-deployment)
        echo "Submitting serving deployment..."
        IMAGE_TAG=$(yq e '.servingDeployment.imageTag' "$CONFIG_FILE")
        ECR_REGISTRY=$(yq e '.servingDeployment.ecrRegistry' "$CONFIG_FILE")
        MLFLOW_TRACKING_URI=$(yq e '.servingDeployment.mlflowTrackingUrl' "$CONFIG_FILE")
        MODEL_VERSION=$(yq e '.servingDeployment.modelVersion' "$CONFIG_FILE")

        argo submit "$ROOT/tools/workflows/serving-deployment-pipeline.yaml" \
            -n argo-workflows \
            -p model-version="$MODEL_VERSION" \
            -p image-tag="$IMAGE_TAG" \
            -p ecr-registry="$ECR_REGISTRY" \
            -p mlflow-tracking-uri="$MLFLOW_TRACKING_URI"
        if [ $? -ne 0 ]; then
            echo "Failed to submit serving deployment."
            return 1 2>/dev/null || exit 1
        fi
        echo "Serving deployment submitted successfully."
        ;;
    *)
        echo "Usage: $0 {--apply-templates|--training-pipeline|--serving-deployment}"
        return 1 2>/dev/null || exit 1
        ;;
esac

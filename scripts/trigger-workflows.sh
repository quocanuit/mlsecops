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
        kubectl apply -f "$ROOT/tools/workflows/templates" --request-timeout=60s
        if [ $? -ne 0 ]; then
            echo "Failed to apply templates."
            return 1 2>/dev/null || exit 1
        fi
        echo "Templates applied successfully."
        ;;
    --training-pipeline)
        echo "Submitting training pipeline..."
        IMAGE_TAG=$(yq -r '.trainingPipeline.imageTag' "$CONFIG_FILE")
        if [ $? -ne 0 ] || [ -z "$IMAGE_TAG" ]; then
            echo "Failed to read imageTag from config."
            return 1 2>/dev/null || exit 1
        fi
        ECR_REGISTRY=$(yq -r '.trainingPipeline.ecrRegistry' "$CONFIG_FILE")
        if [ $? -ne 0 ] || [ -z "$ECR_REGISTRY" ]; then
            echo "Failed to read ecrRegistry from config."
            return 1 2>/dev/null || exit 1
        fi
        MLFLOW_TRACKING_URI=$(yq -r '.trainingPipeline.mlflowTrackingUrl' "$CONFIG_FILE")
        if [ $? -ne 0 ] || [ -z "$MLFLOW_TRACKING_URI" ]; then
            echo "Failed to read mlflowTrackingUrl from config."
            return 1 2>/dev/null || exit 1
        fi
        S3_ARTIFACT_BUCKET=$(yq -r '.trainingPipeline.s3ArtifactBucket' "$CONFIG_FILE")
        if [ $? -ne 0 ] || [ -z "$S3_ARTIFACT_BUCKET" ]; then
            echo "Failed to read s3ArtifactBucket from config."
            return 1 2>/dev/null || exit 1
        fi

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
    --retraining-pipeline)
        echo "Submitting retraining pipeline..."
        IMAGE_TAG=$(yq -r '.retrainingPipeline.imageTag' "$CONFIG_FILE")
        if [ $? -ne 0 ] || [ -z "$IMAGE_TAG" ]; then
            echo "Failed to read imageTag from config."
            return 1 2>/dev/null || exit 1
        fi
        ECR_REGISTRY=$(yq -r '.retrainingPipeline.ecrRegistry' "$CONFIG_FILE")
        if [ $? -ne 0 ] || [ -z "$ECR_REGISTRY" ]; then
            echo "Failed to read ecrRegistry from config."
            return 1 2>/dev/null || exit 1
        fi
        MLFLOW_TRACKING_URI=$(yq -r '.retrainingPipeline.mlflowTrackingUrl' "$CONFIG_FILE")
        if [ $? -ne 0 ] || [ -z "$MLFLOW_TRACKING_URI" ]; then
            echo "Failed to read mlflowTrackingUrl from config."
            return 1 2>/dev/null || exit 1
        fi
        S3_ARTIFACT_BUCKET=$(yq -r '.retrainingPipeline.s3ArtifactBucket' "$CONFIG_FILE")
        if [ $? -ne 0 ] || [ -z "$S3_ARTIFACT_BUCKET" ]; then
            echo "Failed to read s3ArtifactBucket from config."
            return 1 2>/dev/null || exit 1
        fi
        DYNAMODB_TABLE=$(yq -r '.retrainingPipeline.dynamodbTable' "$CONFIG_FILE")
        if [ $? -ne 0 ] || [ -z "$DYNAMODB_TABLE" ]; then
            echo "Failed to read dynamodbTable from config."
            return 1 2>/dev/null || exit 1
        fi
        S3_PRODUCTION_BUCKET=$(yq -r '.retrainingPipeline.s3ProductionBucket' "$CONFIG_FILE")
        if [ $? -ne 0 ] || [ -z "$S3_PRODUCTION_BUCKET" ]; then
            echo "Failed to read s3ProductionBucket from config."
            return 1 2>/dev/null || exit 1
        fi
        MAX_ITEMS=$(yq -r '.retrainingPipeline.maxItems' "$CONFIG_FILE")
        if [ $? -ne 0 ] || [ -z "$MAX_ITEMS" ]; then
            echo "Failed to read maxItems from config."
            return 1 2>/dev/null || exit 1
        fi
        REPLAY_RATIO=$(yq -r '.retrainingPipeline.replayRatio' "$CONFIG_FILE")
        if [ $? -ne 0 ] || [ -z "$REPLAY_RATIO" ]; then
            echo "Failed to read replayRatio from config."
            return 1 2>/dev/null || exit 1
        fi

        argo submit "$ROOT/tools/workflows/retraining-pipeline.yaml" \
            -n argo-workflows \
            -p image-tag="$IMAGE_TAG" \
            -p ecr-registry="$ECR_REGISTRY" \
            -p mlflow-tracking-uri="$MLFLOW_TRACKING_URI" \
            -p s3-artifact-bucket="$S3_ARTIFACT_BUCKET" \
            -p dynamodb-table="$DYNAMODB_TABLE" \
            -p s3-production-bucket="$S3_PRODUCTION_BUCKET" \
            -p max-items="$MAX_ITEMS" \
            -p replay-ratio="$REPLAY_RATIO"
        if [ $? -ne 0 ]; then
            echo "Failed to submit retraining pipeline."
            return 1 2>/dev/null || exit 1
        fi
        echo "Retraining pipeline submitted successfully."
        ;;
    --serving-deployment)
        echo "Submitting serving deployment..."
        IMAGE_TAG=$(yq -r '.servingDeployment.imageTag' "$CONFIG_FILE")
        if [ $? -ne 0 ] || [ -z "$IMAGE_TAG" ]; then
            echo "Failed to read imageTag from config."
            return 1 2>/dev/null || exit 1
        fi
        ECR_REGISTRY=$(yq -r '.servingDeployment.ecrRegistry' "$CONFIG_FILE")
        if [ $? -ne 0 ] || [ -z "$ECR_REGISTRY" ]; then
            echo "Failed to read ecrRegistry from config."
            return 1 2>/dev/null || exit 1
        fi
        MLFLOW_TRACKING_URI=$(yq -r '.servingDeployment.mlflowTrackingUrl' "$CONFIG_FILE")
        if [ $? -ne 0 ] || [ -z "$MLFLOW_TRACKING_URI" ]; then
            echo "Failed to read mlflowTrackingUrl from config."
            return 1 2>/dev/null || exit 1
        fi
        MODEL_VERSION=$(yq -r '.servingDeployment.modelVersion' "$CONFIG_FILE")
        if [ $? -ne 0 ] || [ -z "$MODEL_VERSION" ]; then
            echo "Failed to read modelVersion from config."
            return 1 2>/dev/null || exit 1
        fi

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
        echo "Usage: $0 {--apply-templates|--training-pipeline|--retraining-pipeline|--serving-deployment}"
        return 1 2>/dev/null || exit 1
        ;;
esac

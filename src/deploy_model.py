#!/usr/bin/env python3
import os
import sys
import subprocess
import time
from typing import Optional

try:
    import mlflow
    from mlflow.tracking import MlflowClient
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    print("ERROR: MLflow not available")
    sys.exit(1)


def validate_model_version(model_name: str, version: str) -> bool:
    """Validate that model version exists in MLflow Registry."""
    client = MlflowClient()

    try:
        # Get model version
        model_version = client.get_model_version(model_name, version)

        print(f"Model version found:")
        print(f"  Name: {model_version.name}")
        print(f"  Version: {model_version.version}")
        print(f"  Stage: {model_version.current_stage}")
        print(f"  Status: {model_version.status}")

        if model_version.status != "READY":
            print(f"WARNING: Model status is {model_version.status}, not READY")
            return False

        return True

    except Exception as e:
        print(f"ERROR: Model version not found - {e}")
        return False


def deploy_to_k8s(model_version: str, manifest_path: str = "/app/k8s/model-serving.yaml", deployment_name: str = "fraud-detection-model-server", namespace: str = "mlflow") -> bool:
    print(f"\nDeploying model version {model_version} to Kubernetes...")

    try:
        # Read manifest template
        with open(manifest_path, 'r') as f:
            manifest_content = f.read()

        # Replace placeholders with actual values
        manifest_content = manifest_content.replace(
            "MODEL_VERSION_PLACEHOLDER",
            model_version
        )
        manifest_content = manifest_content.replace(
            "DEPLOYMENT_NAME_PLACEHOLDER",
            deployment_name
        )

        # Write to temp file
        temp_manifest = "/tmp/model-serving-deploy.yaml"
        with open(temp_manifest, 'w') as f:
            f.write(manifest_content)

        print(f"Manifest prepared with version {model_version}")

        # Apply to Kubernetes
        print("Applying to Kubernetes...")
        result = subprocess.run(
            ["kubectl", "apply", "-f", temp_manifest],
            capture_output=True,
            text=True,
            check=True
        )

        print(result.stdout)
        print("Kubernetes resources applied successfully")

        return True

    except subprocess.CalledProcessError as e:
        print(f"ERROR: kubectl apply failed - {e}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        cleanup_on_failure(deployment_name, namespace)
        return False
    except Exception as e:
        print(f"ERROR: Deployment failed - {e}")
        cleanup_on_failure(deployment_name, namespace)
        return False


def wait_for_deployment(namespace: str = "mlflow", deployment_name: str = "fraud-detection-model-server", timeout: int = 300) -> bool:
    print(f"\nWaiting for deployment to be ready (timeout: {timeout}s)...")

    start_time = time.time()

    while time.time() - start_time < timeout:
        try:
            subprocess.run(
                ["kubectl", "rollout", "status", f"deployment/{deployment_name}", "-n", namespace, "--timeout=10s"],
                capture_output=True,
                text=True,
                check=True
            )

            print("Deployment is ready!")
            return True

        except subprocess.CalledProcessError:
            elapsed = int(time.time() - start_time)
            print(f"  Still waiting... ({elapsed}s elapsed)")
            time.sleep(10)

    print(f"ERROR: Deployment not ready after {timeout}s")
    cleanup_on_failure(deployment_name, namespace)
    return False


def health_check(service_url: str = "http://fraud-detection-model-svc.mlflow.svc.cluster.local:8080/health") -> bool:
    """Check if model server is healthy."""

    print(f"\nRunning health check...")
    print(f"   URL: {service_url}")

    try:
        import requests

        response = requests.get(service_url, timeout=10)

        if response.status_code == 200:
            print("Health check passed")
            return True
        else:
            print(f"WARNING: Health check returned {response.status_code}")
            return False

    except ImportError:
        print("WARNING: requests library not available, using curl")
        try:
            result = subprocess.run(
                ["curl", "-f", "-s", service_url],
                capture_output=True,
                text=True,
                timeout=10
            )

            if result.returncode == 0:
                print("Health check passed")
                return True
            else:
                print(f"WARNING: Health check failed")
                return False

        except Exception as e:
            print(f"WARNING: Health check failed - {e}")
            return False
    except Exception as e:
        print(f"WARNING: Health check failed - {e}")
        return False


def cleanup_on_failure(deployment_name: str, namespace: str = "mlflow") -> None:
    print(f"\nCleaning up deployment: {deployment_name} in namespace {namespace}")

    try:
        result = subprocess.run(
            ["kubectl", "delete", "deployment", deployment_name, "-n", namespace, "--ignore-not-found=true"],
            capture_output=True,
            text=True,
            check=False
        )
        if result.returncode == 0:
            print(f"Successfully cleaned up deployment {deployment_name}")
        else:
            print(f"WARNING: Cleanup command exited with code {result.returncode}")
            if result.stderr:
                print(f"stderr: {result.stderr}")
    except Exception as e:
        print(f"WARNING: Failed to cleanup deployment - {e}")


def main():
    # Get parameters from environment
    model_name = os.getenv("MODEL_NAME", "fraud-detection-model")
    model_version = os.getenv("MODEL_VERSION")
    manifest_path = os.getenv("MANIFEST_PATH", "/app/k8s/model-serving.yaml")
    deployment_name = os.getenv("DEPLOYMENT_NAME", "fraud-detection-model-server")
    namespace = os.getenv("NAMESPACE", "mlflow")

    if not model_version:
        print("ERROR: MODEL_VERSION environment variable is required")
        sys.exit(1)

    print("=" * 70)
    print("MODEL DEPLOYMENT WORKFLOW")
    print("=" * 70)
    print(f"Model Name: {model_name}")
    print(f"Model Version: {model_version}")
    print(f"Deployment Name: {deployment_name}")
    print(f"Namespace: {namespace}")
    print(f"Manifest Path: {manifest_path}")

    # Setup MLflow
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        print("ERROR: MLFLOW_TRACKING_URI not set")
        sys.exit(1)

    mlflow.set_tracking_uri(tracking_uri)
    print(f"MLflow URI: {tracking_uri}\n")

    # Step 1: Validate model version
    if not validate_model_version(model_name, model_version):
        print("\nValidation failed")
        sys.exit(1)

    # Step 2: Deploy to K8s
    if not deploy_to_k8s(model_version, manifest_path, deployment_name, namespace):
        print("\nDeployment failed")
        sys.exit(1)

    # Step 3: Wait for deployment
    if not wait_for_deployment(namespace, deployment_name):
        print("\nDeployment not ready")
        sys.exit(1)

    # Step 4: Health check
    health_check()

    print("\n" + "=" * 70)
    print("DEPLOYMENT COMPLETED SUCCESSFULLY")
    print("=" * 70)
    print(f"\nModel {model_name} v{model_version} is now serving!")
    print(f"Endpoint: http://fraud-detection-model-svc.{namespace}.svc.cluster.local:8080/invocations")

    return 0


if __name__ == "__main__":
    sys.exit(main())

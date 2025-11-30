# MLSecOps

## Usage

```bash
aws configure

source scripts/setup.sh

./scripts/trigger-workflows.sh --apply-templates

./scripts/trigger-workflows.sh --training-pipeline

./scripts/trigger-workflows.sh --serving-deployment

./scripts/trigger-workflows.sh --retraining-pipeline

```

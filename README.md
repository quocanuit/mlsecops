# MLSecOps

## Setting up

```bash
aws configure

source scripts/setup.sh

./scripts/trigger-workflows.sh --apply-templates

./scripts/trigger-workflows.sh --training-pipeline

./scripts/trigger-workflows.sh --serving-deployment

```

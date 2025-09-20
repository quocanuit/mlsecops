# MLSecOps

## Quick set up in Colab

```
!git clone https://github.com/quocanuit/mlsecops.git
```
```
import os

os.chdir("/content/mlsecops")

print("Current working directory:", os.getcwd())
```
```
os.environ["AWS_ACCESS_KEY_ID"] ="YOURS"
os.environ["AWS_SECRET_ACCESS_KEY"] = "yours"
os.environ["AWS_REGION"] = "us-east-1"
```
```
!sh set_up_dvc_notebook.sh
```

## How to use DVC

```bash
# Pull data (pulled in set_up script)
dvc pull

# Add/Update data
dvc add # or dvc add <your-changes>

# Commit to git
git add
git commit

# Upload new/changed data to remote
dvc push
```

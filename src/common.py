import os
import subprocess
from pathlib import Path
from dotenv import load_dotenv

# Define project root
ROOT = Path(__file__).resolve().parent.parent

def load_environment():
    env_path = ROOT / ".env"
    if env_path.exists():
        load_dotenv(env_path)
        print(f"Environment variables loaded from: {env_path}")
        return True
    else:
        print(f"Warning: .env file not found at {env_path}")
        return False

def configure_aws_credentials():
    script_path = ROOT / "src" / "script" / "configure_aws_credentials.sh"
    subprocess.run([str(script_path)], check=True)
    os.environ['AWS_PROFILE'] = 'mlsecops_user'
    print(f"AWS_PROFILE set to: {os.environ['AWS_PROFILE']}")

import os

# Sensitive env variables in .env file in form ENV_VAR_NAME=env_var_value

#from dotenv import load_dotenv
#load_dotenv()

# Local environment
LOCAL_DATA_DIR = os.getenv("LOCAL_DATA_DIR")

# Paperspace Config
x_api_key = os.getenv("X_API_KEY")
HEADERS = {"x-api-key": x_api_key}

# Docker Config
REGISTRY_USER = os.getenv("REGISTRY_USER")
REGISTRY_PASSWORD = os.getenv("REGISTRY_PASSWORD")

# S3 Config
S3_BUCKET = os.getenv("S3_BUCKET")

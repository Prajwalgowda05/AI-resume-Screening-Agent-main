"""Application configuration module using python-dotenv and os.environ."""

import os
from pathlib import Path

# Safely attempt to load .env if python-dotenv is installed
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent

# Environment Settings
APP_ENV = os.environ.get("APP_ENV", "development")
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")

# Directories
DATA_DIR = Path(os.environ.get("DATA_DIR", BASE_DIR / "data"))
RESUMES_DIR = Path(os.environ.get("RESUMES_DIR", DATA_DIR / "resumes"))
JOB_DESCRIPTIONS_DIR = Path(
    os.environ.get("JOB_DESCRIPTIONS_DIR", DATA_DIR / "job_descriptions")
)
OUTPUTS_DIR = Path(os.environ.get("OUTPUTS_DIR", BASE_DIR / "outputs"))

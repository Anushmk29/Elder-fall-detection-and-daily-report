import sys
from pathlib import Path

if "pytest" not in sys.modules:  # tests set their own environment (no API keys)
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")  # never overrides; uvicorn --env-file still works

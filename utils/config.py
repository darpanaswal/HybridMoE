import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).parent.parent
dotenv_path = BASE_DIR / ".env"

load_dotenv(dotenv_path=dotenv_path)

openai_token = os.getenv("OPENAI_API_KEY")
hf_token = os.getenv("HUGGINGFACE_API_KEY")
wandb_token = os.getenv("WANDB_API_KEY")
google_token = os.getenv("GOOGLE_API_KEY")
mistral_token = os.getenv("MISTRAL_API_KEY")

if not openai_token or not hf_token or not wandb_token:
    raise ValueError("API keys are not set in environment variables")

DATA_PATH = BASE_DIR / "data"
MODEL_PATH = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"
import os
from pathlib import Path
from dotenv import load_dotenv

def get_file_path(marker = ".env") -> Path:
    current = Path(__file__).resolve().parent
    while current != current.parent:
        if(current/marker).exists():
            return current/marker
        current = current.parent
    raise FileNotFoundError(f"Could not find {marker} in {current.parent}")

load_dotenv(dotenv_path=get_file_path())

class Settings:
    def __init__(self):
        self.api_key = os.getenv("API_KEY")
        self.base_url = os.getenv("BASE_URL_CHAT")
        self.model = os.getenv("MODEL")

settings = Settings()
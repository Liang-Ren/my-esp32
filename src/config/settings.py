import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parents[2] / ".env", override=True)


class Settings:
    # OpenAI
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    # Mem0
    MEM0_API_KEY: str = os.getenv("MEM0_API_KEY", "")
    MEM0_SERVER_URL: str = os.getenv("MEM0_SERVER_URL", "")
    MEM0_USER_ID_PREFIX: str = os.getenv("MEM0_USER_ID_PREFIX", "xiaozhi_")

    # WebSocket server
    WS_HOST: str = os.getenv("WS_HOST", "0.0.0.0")
    WS_PORT: int = int(os.getenv("WS_PORT", "8001"))
    MAX_LISTEN_FRAMES: int = int(os.getenv("MAX_LISTEN_FRAMES", "25"))
    SILENCE_TIMEOUT: float = float(os.getenv("SILENCE_TIMEOUT", "1.5"))


settings = Settings()

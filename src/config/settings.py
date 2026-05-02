import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parents[2] / ".env", override=True)


class Settings:
    # OpenAI
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    LLM_TIMEOUT: float = float(os.getenv("LLM_TIMEOUT", "30.0"))

    # Mem0
    MEM0_API_KEY: str = os.getenv("MEM0_API_KEY", "")
    MEM0_SERVER_URL: str = os.getenv("MEM0_SERVER_URL", "")
    MEM0_USER_ID_PREFIX: str = os.getenv("MEM0_USER_ID_PREFIX", "xiaozhi_")
    MEMORY_TIMEOUT: float = float(os.getenv("MEMORY_TIMEOUT", "5.0"))

    # WebSocket server
    WS_HOST: str = os.getenv("WS_HOST", "0.0.0.0")
    WS_PORT: int = int(os.getenv("WS_PORT", "8001"))
    WS_PING_INTERVAL: int = int(os.getenv("WS_PING_INTERVAL", "20"))
    WS_PING_TIMEOUT: int = int(os.getenv("WS_PING_TIMEOUT", "20"))
    MAX_LISTEN_FRAMES: int = int(os.getenv("MAX_LISTEN_FRAMES", "50"))
    SILENCE_TIMEOUT: float = float(os.getenv("SILENCE_TIMEOUT", "1.5"))

    # Health server (HTTP, separate port from WebSocket)
    HEALTH_PORT: int = int(os.getenv("HEALTH_PORT", "8002"))

    # Voice response
    MAX_VOICE_REPLY_CHARS: int = int(os.getenv("MAX_VOICE_REPLY_CHARS", "300"))

    def validate(self) -> None:
        """Raise ValueError listing every configuration problem found."""
        errors: list[str] = []

        if not self.OPENAI_API_KEY:
            errors.append("OPENAI_API_KEY is required but not set")
        if not self.OPENAI_MODEL:
            errors.append("OPENAI_MODEL is required but not set")

        if not (1 <= self.WS_PORT <= 65535):
            errors.append(f"WS_PORT={self.WS_PORT} is outside 1–65535")
        if not (1 <= self.HEALTH_PORT <= 65535):
            errors.append(f"HEALTH_PORT={self.HEALTH_PORT} is outside 1–65535")
        if self.WS_PORT == self.HEALTH_PORT:
            errors.append(
                f"WS_PORT and HEALTH_PORT must be different (both are {self.WS_PORT})"
            )

        if self.MEMORY_TIMEOUT <= 0:
            errors.append(f"MEMORY_TIMEOUT must be > 0 (got {self.MEMORY_TIMEOUT})")
        if self.LLM_TIMEOUT <= 0:
            errors.append(f"LLM_TIMEOUT must be > 0 (got {self.LLM_TIMEOUT})")
        if self.SILENCE_TIMEOUT <= 0:
            errors.append(f"SILENCE_TIMEOUT must be > 0 (got {self.SILENCE_TIMEOUT})")
        if self.MAX_LISTEN_FRAMES < 1:
            errors.append(
                f"MAX_LISTEN_FRAMES must be >= 1 (got {self.MAX_LISTEN_FRAMES})"
            )
        if self.MAX_VOICE_REPLY_CHARS < 10:
            errors.append(
                f"MAX_VOICE_REPLY_CHARS must be >= 10 (got {self.MAX_VOICE_REPLY_CHARS})"
            )

        if errors:
            raise ValueError(
                "Bad configuration:\n" + "\n".join(f"  • {e}" for e in errors)
            )


settings = Settings()

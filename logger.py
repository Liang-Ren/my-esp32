import json
import uuid
import logging
from datetime import datetime
from pathlib import Path

LOG_FILE = Path(__file__).parent / "server.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
_logger = logging.getLogger("xiaozhi")


def new_request_id() -> str:
    return str(uuid.uuid4())[:8]


def log(msg: str):
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    _logger.info(line)


def log_request(
    request_id: str,
    device_id: str,
    user_text: str,
    response_text: str,
    model: str,
    latency_ms: float,
    usage: dict = None,
    error: str = None,
):
    entry = {
        "ts": datetime.utcnow().isoformat(),
        "req": request_id,
        "device": device_id,
        "user": user_text[:120],
        "resp": response_text[:120],
        "model": model,
        "latency_ms": round(latency_ms),
    }
    if usage:
        entry["tokens"] = usage.get("total_tokens")
    if error:
        entry["error"] = error
    _logger.info(json.dumps(entry, ensure_ascii=False))
"""
Structured logger for the Xiaozhi server.

Functions:
    log(msg)                       — general info line
    log_step(req_id, step, detail) — one pipeline step with optional timing
    log_request(...)               — end-of-request JSON summary line
    new_request_id()               — 8-char hex ID
"""
import json
import uuid
import logging as _stdlib
from datetime import datetime, timezone
from pathlib import Path

_LOG_FILE = Path(__file__).parents[2] / "server.log"

_stdlib.basicConfig(
    level=_stdlib.INFO,
    format="%(message)s",
    handlers=[
        _stdlib.FileHandler(_LOG_FILE, encoding="utf-8"),
        _stdlib.StreamHandler(),
    ],
)
_logger = _stdlib.getLogger("xiaozhi.src")


def new_request_id() -> str:
    return uuid.uuid4().hex[:8]


def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


def log(msg: str) -> None:
    _logger.info(f"[{_ts()}] {msg}")


def log_step(
    request_id: str,
    step: str,
    detail: str,
    ms: float | None = None,
) -> None:
    """
    Emit one structured step line.

    Format:
        [HH:MM:SS]   [req_id] [step  ] detail (42ms)
    """
    suffix = f" ({ms}ms)" if ms is not None else ""
    _logger.info(f"[{_ts()}]   [{request_id}] [{step:<7s}] {detail}{suffix}")


def log_request(
    request_id: str,
    device_id: str,
    user_text: str,
    response_text: str,
    model: str,
    metrics: dict,
    error: str | None = None,
) -> None:
    """
    Emit a single JSON line summarising the completed request.

    metrics should contain at minimum: parse_ms, memory_ms, llm_ms, total_ms.
    Any extra keys (asr_ms, tts_ms, tokens, …) are included as-is.
    """
    entry: dict = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "req": request_id,
        "device": device_id,
        "user": user_text[:120],
        "resp": response_text[:120],
        "model": model,
    }
    entry.update(metrics)
    if error:
        entry["error"] = error
    _logger.info(json.dumps(entry, ensure_ascii=False))

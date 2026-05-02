"""
Structured logger for the Xiaozhi server.

Functions:
    log(msg)                       — general info line
    log_step(req_id, step, detail) — one pipeline step with optional timing
    log_request(...)               — end-of-request JSON summary line
    new_request_id()               — 8-char hex ID
    redact(secret)                 — register a secret to strip from all log output
"""
import json
import uuid
import logging as _stdlib
from datetime import datetime, timezone
from pathlib import Path

_LOG_FILE = Path(__file__).parents[2] / "server.log"


class _RedactFilter(_stdlib.Filter):
    """Strip registered secrets from every log record before it is emitted."""

    _secrets: list[str] = []   # class-level: shared across all instances

    @classmethod
    def register(cls, secret: str) -> None:
        """Add a secret to redact. Safe to call multiple times with the same value."""
        if secret and secret not in cls._secrets:
            cls._secrets.append(secret)

    def filter(self, record: _stdlib.LogRecord) -> bool:
        if not self._secrets:
            return True
        try:
            msg = record.getMessage()
        except Exception:
            return True
        replaced = False
        for s in self._secrets:
            if s in msg:
                msg = msg.replace(s, "[REDACTED]")
                replaced = True
        if replaced:
            record.msg = msg
            record.args = ()   # args already baked into msg
        return True


_redact_filter = _RedactFilter()

_stdlib.basicConfig(
    level=_stdlib.INFO,
    format="%(message)s",
    handlers=[
        _stdlib.FileHandler(_LOG_FILE, encoding="utf-8"),
        _stdlib.StreamHandler(),
    ],
)
_logger = _stdlib.getLogger("xiaozhi.src")
_logger.addFilter(_redact_filter)


def redact(secret: str) -> None:
    """Register a secret string to strip from all future log output."""
    _RedactFilter.register(secret)


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

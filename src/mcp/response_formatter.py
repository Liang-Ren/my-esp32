"""
ResponseFormatter — the last step before bytes go to ESP32.

Responsibilities:
  1. Sanitize LLM text for voice (strip markdown, enforce length)
  2. Generate Opus frames via tts_service

FormattedResponse is the only thing that crosses the boundary between
the AI pipeline and the ESP32 send path.
"""
import re
import sys
from dataclasses import dataclass
from pathlib import Path

# tts_service lives in the project root, not under src/
_ROOT = Path(__file__).parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import tts_service

# Patterns that don't translate well to voice
_STRIP_MD = re.compile(r"[*_`#~|>\\]")
_MULTI_SPACE = re.compile(r" {2,}")

# Hard cap: edge-tts is slow on very long strings and ESP32 buffers are finite
MAX_CHARS = 300


@dataclass
class FormattedResponse:
    text: str              # sanitized, voice-safe text
    opus_frames: list[bytes]  # Opus frames ready to stream to ESP32


class ResponseFormatter:
    """
    Converts a raw LLM reply string into a FormattedResponse.

    Usage:
        formatter = ResponseFormatter()
        result = await formatter.format(raw_llm_text)
        # result.text      — logged, sent as TTS sentence JSON
        # result.opus_frames — streamed as binary WebSocket frames
    """

    async def format(self, raw_text: str) -> FormattedResponse:
        text = _sanitize(raw_text)
        frames = await tts_service.generate(text)
        return FormattedResponse(text=text, opus_frames=frames)


def _sanitize(text: str) -> str:
    """Remove markdown and trim to voice-safe length."""
    text = _STRIP_MD.sub("", text)
    text = _MULTI_SPACE.sub(" ", text).strip()
    if len(text) > MAX_CHARS:
        # Break cleanly at the last sentence boundary within the limit
        truncated = text[:MAX_CHARS]
        last_period = truncated.rfind("。")
        text = (truncated[:last_period + 1] if last_period > 0 else truncated) + "…"
    return text

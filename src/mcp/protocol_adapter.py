"""
Isolates all ESP32 Xiaozhi protocol v1 wire formats.
Nothing outside this module should know about raw message field names.
"""
import json
import struct
from dataclasses import dataclass


# ── Inbound message models ────────────────────────────────────────────────────

@dataclass
class HelloMsg:
    session_id: str
    version: int


@dataclass
class ListenMsg:
    state: str   # "start" | "stop" | "detect"
    mode: str    # "realtime" | "auto" | "manual"


# ── Adapter ───────────────────────────────────────────────────────────────────

class ProtocolAdapter:
    """Parses inbound ESP32 JSON and builds outbound JSON/binary frames."""

    # Audio constants (must match tts_service)
    SAMPLE_RATE = 16000
    CHANNELS = 1
    FRAME_DURATION_MS = 60

    def parse_hello(self, msg: dict) -> HelloMsg:
        return HelloMsg(
            session_id=msg.get("session_id", ""),
            version=int(msg.get("version", 1)),
        )

    def parse_listen(self, msg: dict) -> ListenMsg:
        return ListenMsg(
            state=msg.get("state", ""),
            mode=msg.get("mode", "auto"),
        )

    def build_hello(self, session_id: str) -> str:
        return json.dumps({
            "type": "hello",
            "transport": "websocket",
            "session_id": session_id,
            "audio_params": {
                "format": "opus",
                "sample_rate": self.SAMPLE_RATE,
                "channels": self.CHANNELS,
                "frame_duration": self.FRAME_DURATION_MS,
            },
        })

    def build_tts_start(self, session_id: str) -> str:
        return json.dumps({"type": "tts", "state": "start", "session_id": session_id})

    def build_tts_sentence(self, text: str, session_id: str) -> str:
        return json.dumps({
            "type": "tts",
            "state": "sentence_start",
            "text": text,
            "session_id": session_id,
        })

    def build_tts_stop(self, session_id: str) -> str:
        return json.dumps({"type": "tts", "state": "stop", "session_id": session_id})

    @staticmethod
    def wrap_opus_frame(opus_data: bytes, version: int) -> bytes:
        """Version-aware binary frame wrapping (must stay in sync with tts_service.make_frame)."""
        if version == 2:
            return struct.pack(">HHHII", 2, 0, 0, 0, len(opus_data)) + opus_data
        if version == 3:
            return bytes([0, 0]) + struct.pack(">H", len(opus_data)) + opus_data
        return opus_data  # v1: raw

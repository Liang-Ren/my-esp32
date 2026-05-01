"""
Shared fixtures for all test modules.
No real API keys, network calls, or file I/O required.
"""
import sys
import json
import asyncio
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

# Ensure project root is importable from test files
sys.path.insert(0, str(Path(__file__).parents[1]))

from src.mcp.session_manager import Session
from src.mcp.response_formatter import FormattedResponse


# ── Sessions / devices ────────────────────────────────────────────────────────

@pytest.fixture
def session() -> Session:
    """Standard single-device session."""
    return Session(
        connection_id="conn-001",
        device_id="device_10_0_0_1",
        user_id="xiaozhi_device_10_0_0_1",
        session_id="sess-abc123",
        proto_version=1,
        listen_mode="auto",
    )


@pytest.fixture
def session_a() -> Session:
    return Session(
        connection_id="conn-a",
        device_id="device_A",
        user_id="xiaozhi_device_A",
        session_id="sess-aaaa",
        proto_version=1,
        listen_mode="auto",
    )


@pytest.fixture
def session_b() -> Session:
    return Session(
        connection_id="conn-b",
        device_id="device_B",
        user_id="xiaozhi_device_B",
        session_id="sess-bbbb",
        proto_version=1,
        listen_mode="auto",
    )


# ── WebSocket stub ────────────────────────────────────────────────────────────

class FakeWebSocket:
    """
    Minimal WebSocket stub.
    Records every send() call so tests can inspect what was sent.

    Separate lists for text (JSON control frames) and binary (Opus frames).
    """
    def __init__(self, addr=("10.0.0.1", 12345)):
        self.remote_address = addr
        self._sent: list = []

    async def send(self, data) -> None:
        self._sent.append(data)

    @property
    def json_frames(self) -> list[dict]:
        return [json.loads(m) for m in self._sent if isinstance(m, str)]

    @property
    def binary_frames(self) -> list[bytes]:
        return [m for m in self._sent if isinstance(m, bytes)]


@pytest.fixture
def fake_ws() -> FakeWebSocket:
    return FakeWebSocket()


# ── Mem0 mock ─────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_mem0() -> AsyncMock:
    """Mem0 client mock that returns no memories by default."""
    m = AsyncMock()
    m.search.return_value = []
    m.add.return_value = None
    m.get_all.return_value = []
    return m


# ── SQLite stub ───────────────────────────────────────────────────────────────

class FakeSqlite:
    """
    Pure in-memory replacement for memory.Memory.
    Satisfies the same interface without touching any files.
    """
    def __init__(self):
        self._messages: dict[str, list[dict]] = {}
        self._long_term: dict[str, dict] = {}

    def add_message(self, device_id: str, role: str, content: str) -> None:
        self._messages.setdefault(device_id, []).append(
            {"role": role, "content": content}
        )

    def get_recent(self, device_id: str, limit: int = 10) -> list[dict]:
        return self._messages.get(device_id, [])[-limit:]

    def get_long_term(self, device_id: str) -> dict:
        return self._long_term.get(
            device_id, {"summary": "", "facts": [], "preferences": {}}
        )

    def update_long_term(
        self,
        device_id: str,
        summary=None,
        facts=None,
        preferences=None,
    ) -> None:
        existing = self.get_long_term(device_id)
        self._long_term[device_id] = {
            "summary": summary if summary is not None else existing["summary"],
            "facts": facts if facts is not None else existing["facts"],
            "preferences": (
                preferences if preferences is not None else existing["preferences"]
            ),
        }


@pytest.fixture
def fake_sqlite() -> FakeSqlite:
    return FakeSqlite()


# ── LLM mock ─────────────────────────────────────────────────────────────────

MOCK_REPLY = "这是一个测试回复。"

@pytest.fixture
def mock_llm() -> AsyncMock:
    """OpenAI client mock that returns a fixed reply with no error."""
    m = AsyncMock()
    m.model = "gpt-4o-mini-test"
    m.generateResponse.return_value = (
        MOCK_REPLY,
        {
            "model": "gpt-4o-mini-test",
            "total_tokens": 42,
            "latency_ms": 50,
            "response_id": "resp_test123",
        },
    )
    return m


# ── Pre-built FormattedResponse ───────────────────────────────────────────────

FAKE_OPUS_FRAMES = [b"\x00\x01\x02", b"\x03\x04\x05"]

@pytest.fixture
def fake_formatted() -> FormattedResponse:
    return FormattedResponse(text=MOCK_REPLY, opus_frames=FAKE_OPUS_FRAMES)

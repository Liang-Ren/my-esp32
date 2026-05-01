"""
Test 2 — Memory lookup is called before OpenAI
Test 3 — OpenAI prompt includes relevant memory
Test 4 — Assistant reply is stored back to memory
Test 6 — OpenAI failure returns fallback response
Test 7 — Response format remains ESP32-compatible (full send sequence)

All tests drive _process_turn() directly.
asr.transcribe and tts_service.generate are patched so no hardware is needed.
"""
import asyncio
import json
from unittest.mock import AsyncMock, patch

import pytest

import src.mcp.gateway as gw
from src.mcp.response_formatter import FormattedResponse
from src.memory.memory_service import MemoryService
from src.ai.openai_client import FALLBACK_RESPONSE

from tests.conftest import FakeSqlite, FAKE_OPUS_FRAMES, MOCK_REPLY


# ── Shared setup ──────────────────────────────────────────────────────────────

USER_TEXT = "我想了解一下温哥华的天气。"
FAKE_FORMATTED = FormattedResponse(text=MOCK_REPLY, opus_frames=FAKE_OPUS_FRAMES)
FAKE_FALLBACK = FormattedResponse(text="我没听清楚，请再说一遍。", opus_frames=[b"\xff\xff"])


def _make_service(mem0, sqlite=None) -> MemoryService:
    return MemoryService(mem0_client=mem0, sqlite_memory=sqlite or FakeSqlite())


async def _run_turn(fake_ws, session, memory_svc, mock_llm) -> None:
    """
    Call _process_turn with patches that prevent any real I/O.

    Patches applied every call:
      - asr.transcribe         → returns USER_TEXT
      - _formatter.format      → returns FAKE_FORMATTED
      - gateway._fallback_resp → set to FAKE_FALLBACK
      - tts_service.FRAME_DURATION_MS → 0  (skip inter-frame sleep)
    """
    with (
        patch("asr.transcribe", new_callable=AsyncMock) as mock_asr,
        patch.object(
            gw._formatter, "format",
            new_callable=AsyncMock,
            return_value=FAKE_FORMATTED,
        ),
        patch("src.mcp.gateway.tts_service.FRAME_DURATION_MS", 0),
    ):
        mock_asr.return_value = USER_TEXT
        gw._fallback_resp = FAKE_FALLBACK

        await gw._process_turn(fake_ws, [b"\x00" * 20], session, memory_svc, mock_llm)

        # Flush any ensure_future tasks (store step)
        await asyncio.sleep(0)


# ── Test 2: Memory called before OpenAI ───────────────────────────────────────

class TestMemoryCalledBeforeLLM:
    async def test_memory_lookup_precedes_llm_call(
        self, fake_ws, session, mock_llm
    ):
        call_order: list[str] = []

        mem0 = AsyncMock()
        mem0.search.side_effect = (
            lambda *a, **kw: call_order.append("memory") or []
        )
        mem0.add.return_value = None

        async def tracking_generate(input_messages, context="", **kw):
            call_order.append("llm")
            return MOCK_REPLY, {"model": "test", "total_tokens": 10}

        mock_llm.generateResponse = tracking_generate

        svc = _make_service(mem0)
        await _run_turn(fake_ws, session, svc, mock_llm)

        assert "memory" in call_order, "Memory search was never called"
        assert "llm" in call_order, "LLM was never called"
        assert call_order.index("memory") < call_order.index("llm"), (
            f"Memory should be called before LLM, got: {call_order}"
        )

    async def test_memory_called_even_for_short_queries(
        self, fake_ws, session, mock_llm
    ):
        mem0 = AsyncMock()
        mem0.search.return_value = []

        svc = _make_service(mem0)
        await _run_turn(fake_ws, session, svc, mock_llm)

        mem0.search.assert_awaited_once()


# ── Test 3: Prompt includes relevant memory ────────────────────────────────────

class TestMemoryInjectedIntoPrompt:
    async def test_relevant_memory_appears_in_instructions(
        self, fake_ws, session, mock_llm
    ):
        KNOWN_MEMORY = "user lives in Vancouver and loves hiking"
        captured: dict = {}

        mem0 = AsyncMock()
        mem0.search.return_value = [{"memory": KNOWN_MEMORY}]
        mem0.add.return_value = None

        async def capture_generate(input_messages, context="", **kw):
            captured["context"] = context
            captured["input_messages"] = input_messages
            return MOCK_REPLY, {"model": "test", "total_tokens": 10}

        mock_llm.generateResponse = capture_generate

        svc = _make_service(mem0)
        await _run_turn(fake_ws, session, svc, mock_llm)

        assert "context" in captured, "generateResponse was never called"
        assert KNOWN_MEMORY in captured["context"], (
            f"Expected {KNOWN_MEMORY!r} in instructions.\n"
            f"Got: {captured.get('context', '')[:300]}"
        )

    async def test_memory_summary_appears_in_instructions(
        self, fake_ws, session, mock_llm
    ):
        SUMMARY = "long-time Vancouver resident"
        captured: dict = {}

        sqlite = FakeSqlite()
        sqlite.update_long_term(session.device_id, summary=SUMMARY)

        mem0 = AsyncMock()
        mem0.search.return_value = []

        async def capture_generate(input_messages, context="", **kw):
            captured["context"] = context
            return MOCK_REPLY, {"model": "test", "total_tokens": 10}

        mock_llm.generateResponse = capture_generate

        svc = _make_service(mem0, sqlite)
        await _run_turn(fake_ws, session, svc, mock_llm)

        assert SUMMARY in captured.get("context", ""), (
            "SQLite summary should be injected into system instructions"
        )

    async def test_no_memory_gives_clean_base_prompt(
        self, fake_ws, session, mock_llm
    ):
        captured: dict = {}

        mem0 = AsyncMock()
        mem0.search.return_value = []

        async def capture_generate(input_messages, context="", **kw):
            captured["context"] = context
            return MOCK_REPLY, {"model": "test", "total_tokens": 10}

        mock_llm.generateResponse = capture_generate

        svc = _make_service(mem0)
        await _run_turn(fake_ws, session, svc, mock_llm)

        assert "【近期记忆】" not in captured.get("context", "")
        assert "【用户背景】" not in captured.get("context", "")


# ── Test 4: Reply stored back to memory ───────────────────────────────────────

class TestInteractionStored:
    async def test_addInteraction_called_with_user_and_assistant_text(
        self, fake_ws, session, mock_llm
    ):
        mem0 = AsyncMock()
        mem0.search.return_value = []
        mem0.add.return_value = None

        svc = _make_service(mem0)
        await _run_turn(fake_ws, session, svc, mock_llm)

        mem0.add.assert_awaited_once()
        args, kwargs = mem0.add.call_args
        messages = args[0]
        user_id = kwargs.get("user_id") or args[1]

        roles = {m["role"]: m["content"] for m in messages}
        assert roles.get("user") == USER_TEXT
        assert roles.get("assistant") == MOCK_REPLY   # from mock_llm
        assert user_id == session.user_id

    async def test_sqlite_history_written_after_turn(
        self, fake_ws, session, mock_llm
    ):
        mem0 = AsyncMock()
        mem0.search.return_value = []
        mem0.add.return_value = None
        sqlite = FakeSqlite()

        svc = _make_service(mem0, sqlite)
        await _run_turn(fake_ws, session, svc, mock_llm)

        history = svc.get_history(session.device_id)
        contents = {m["content"] for m in history}
        assert USER_TEXT in contents
        assert MOCK_REPLY in contents


# ── Test 6: OpenAI failure → fallback response ────────────────────────────────

class TestOpenAIFailure:
    async def test_error_in_usage_sends_fallback_tts(
        self, fake_ws, session, mock_llm
    ):
        """When generateResponse returns an error key, TTS uses FALLBACK_TEXT."""
        mem0 = AsyncMock()
        mem0.search.return_value = []
        mock_llm.generateResponse.return_value = (
            FALLBACK_RESPONSE,
            {"error": "API 503: service unavailable", "latency_ms": 100},
        )

        svc = _make_service(mem0)
        await _run_turn(fake_ws, session, svc, mock_llm)

        json_frames = fake_ws.json_frames
        tts_frames = [f for f in json_frames if f.get("type") == "tts"]
        assert tts_frames, "No TTS frames sent at all"
        # Fallback text (not the LLM reply) should be the sentence text
        sentence_frame = next(
            (f for f in tts_frames if f.get("state") == "sentence_start"), None
        )
        assert sentence_frame is not None
        # The gateway sends FALLBACK_TEXT (Chinese), not the English FALLBACK_RESPONSE
        assert sentence_frame["text"] != MOCK_REPLY

    async def test_interaction_not_stored_on_llm_error(
        self, fake_ws, session, mock_llm
    ):
        """When the LLM errors, nothing is added to Mem0."""
        mem0 = AsyncMock()
        mem0.search.return_value = []
        mock_llm.generateResponse.return_value = (
            FALLBACK_RESPONSE,
            {"error": "timeout after 30.0s", "latency_ms": 30000},
        )

        svc = _make_service(mem0)
        await _run_turn(fake_ws, session, svc, mock_llm)

        mem0.add.assert_not_awaited()

    async def test_empty_asr_sends_fallback_without_calling_llm(
        self, fake_ws, session, mock_llm
    ):
        """Silence / empty transcription → fallback without hitting OpenAI."""
        mem0 = AsyncMock()
        gw._fallback_resp = FAKE_FALLBACK

        with (
            patch("asr.transcribe", new_callable=AsyncMock) as mock_asr,
            patch("src.mcp.gateway.tts_service.FRAME_DURATION_MS", 0),
        ):
            mock_asr.return_value = ""  # empty — no speech

            svc = _make_service(mem0)
            await gw._process_turn(
                fake_ws, [b"\x00" * 10], session, svc, mock_llm
            )

        mock_llm.generateResponse.assert_not_awaited()
        assert fake_ws.json_frames, "No TTS frames sent for empty ASR"


# ── Test 7: Full ESP32-compatible send sequence ────────────────────────────────

class TestESP32ResponseFormat:
    async def test_tts_sequence_order(self, fake_ws, session, mock_llm):
        """Verify: tts-start → sentence_start → [binary] → tts-stop."""
        mem0 = AsyncMock()
        mem0.search.return_value = []
        svc = _make_service(mem0)

        await _run_turn(fake_ws, session, svc, mock_llm)

        frames = fake_ws._sent
        assert frames, "Nothing was sent to the device"

        # Find positions of control frames
        tts_frames = [
            (i, json.loads(f))
            for i, f in enumerate(frames)
            if isinstance(f, str) and json.loads(f).get("type") == "tts"
        ]
        binary_positions = [i for i, f in enumerate(frames) if isinstance(f, bytes)]

        states = [d["state"] for _, d in tts_frames]
        assert "start" in states
        assert "sentence_start" in states
        assert "stop" in states

        start_pos = next(i for i, d in tts_frames if d["state"] == "start")
        stop_pos = next(i for i, d in tts_frames if d["state"] == "stop")
        sentence_pos = next(i for i, d in tts_frames if d["state"] == "sentence_start")

        assert start_pos < sentence_pos < stop_pos, (
            "Expected: start < sentence_start < stop"
        )
        for bp in binary_positions:
            assert start_pos < bp < stop_pos, (
                f"Binary frame at pos {bp} is outside start/stop window"
            )

    async def test_tts_frames_carry_correct_session_id(self, fake_ws, session, mock_llm):
        mem0 = AsyncMock()
        mem0.search.return_value = []
        svc = _make_service(mem0)

        await _run_turn(fake_ws, session, svc, mock_llm)

        for frame in fake_ws.json_frames:
            if frame.get("type") == "tts":
                assert frame.get("session_id") == session.session_id

    async def test_binary_frames_are_raw_opus_for_v1(self, fake_ws, session, mock_llm):
        """Proto v1: binary frames must equal the raw Opus bytes (no header)."""
        assert session.proto_version == 1
        mem0 = AsyncMock()
        mem0.search.return_value = []
        svc = _make_service(mem0)

        await _run_turn(fake_ws, session, svc, mock_llm)

        assert fake_ws.binary_frames, "No binary Opus frames sent"
        for frame in fake_ws.binary_frames:
            assert frame in FAKE_OPUS_FRAMES, (
                "v1 binary frame should equal the raw Opus bytes from FormattedResponse"
            )

    async def test_response_formatter_strips_markdown(self):
        """Standalone formatter test — markdown must not reach the ESP32."""
        from src.mcp.response_formatter import _sanitize

        raw = "**Hello** `code` | table | _italic_ # header"
        clean = _sanitize(raw)

        for char in ("*", "`", "|", "_", "#"):
            assert char not in clean, f"Char {char!r} not stripped from {clean!r}"

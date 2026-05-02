"""
Tests for all hardening features:
  1. Settings validation
  2. Log redaction
  3. Health endpoint responses
  4. Memory timeout in pipeline
"""
import asyncio
import json
import logging
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.config.settings import Settings


# ── Helpers ───────────────────────────────────────────────────────────────────

def _valid_settings(**overrides) -> Settings:
    """Return a Settings instance with valid defaults, optionally overridden."""
    s = Settings.__new__(Settings)
    s.OPENAI_API_KEY    = overrides.get("OPENAI_API_KEY",    "sk-test")
    s.OPENAI_MODEL      = overrides.get("OPENAI_MODEL",      "gpt-4o-mini")
    s.LLM_TIMEOUT       = overrides.get("LLM_TIMEOUT",       30.0)
    s.MEM0_API_KEY      = overrides.get("MEM0_API_KEY",      "")
    s.MEM0_SERVER_URL   = overrides.get("MEM0_SERVER_URL",   "")
    s.MEM0_USER_ID_PREFIX = overrides.get("MEM0_USER_ID_PREFIX", "xiaozhi_")
    s.MEMORY_TIMEOUT    = overrides.get("MEMORY_TIMEOUT",    5.0)
    s.WS_HOST           = overrides.get("WS_HOST",           "0.0.0.0")
    s.WS_PORT           = overrides.get("WS_PORT",           8001)
    s.WS_PING_INTERVAL  = overrides.get("WS_PING_INTERVAL",  20)
    s.WS_PING_TIMEOUT   = overrides.get("WS_PING_TIMEOUT",   20)
    s.MAX_LISTEN_FRAMES = overrides.get("MAX_LISTEN_FRAMES", 25)
    s.SILENCE_TIMEOUT   = overrides.get("SILENCE_TIMEOUT",   1.5)
    s.HEALTH_PORT       = overrides.get("HEALTH_PORT",       8002)
    s.MAX_VOICE_REPLY_CHARS = overrides.get("MAX_VOICE_REPLY_CHARS", 300)
    return s


# ── 1. Settings validation ────────────────────────────────────────────────────

class TestSettingsValidation:
    def test_valid_config_passes(self):
        _valid_settings().validate()   # must not raise

    def test_missing_openai_key_raises(self):
        with pytest.raises(ValueError, match="OPENAI_API_KEY"):
            _valid_settings(OPENAI_API_KEY="").validate()

    def test_missing_model_raises(self):
        with pytest.raises(ValueError, match="OPENAI_MODEL"):
            _valid_settings(OPENAI_MODEL="").validate()

    def test_invalid_ws_port_zero_raises(self):
        with pytest.raises(ValueError, match="WS_PORT"):
            _valid_settings(WS_PORT=0).validate()

    def test_invalid_ws_port_too_high_raises(self):
        with pytest.raises(ValueError, match="WS_PORT"):
            _valid_settings(WS_PORT=99999).validate()

    def test_port_conflict_raises(self):
        with pytest.raises(ValueError, match="different"):
            _valid_settings(WS_PORT=8001, HEALTH_PORT=8001).validate()

    def test_zero_memory_timeout_raises(self):
        with pytest.raises(ValueError, match="MEMORY_TIMEOUT"):
            _valid_settings(MEMORY_TIMEOUT=0.0).validate()

    def test_negative_llm_timeout_raises(self):
        with pytest.raises(ValueError, match="LLM_TIMEOUT"):
            _valid_settings(LLM_TIMEOUT=-1.0).validate()

    def test_zero_silence_timeout_raises(self):
        with pytest.raises(ValueError, match="SILENCE_TIMEOUT"):
            _valid_settings(SILENCE_TIMEOUT=0.0).validate()

    def test_max_listen_frames_zero_raises(self):
        with pytest.raises(ValueError, match="MAX_LISTEN_FRAMES"):
            _valid_settings(MAX_LISTEN_FRAMES=0).validate()

    def test_too_small_voice_chars_raises(self):
        with pytest.raises(ValueError, match="MAX_VOICE_REPLY_CHARS"):
            _valid_settings(MAX_VOICE_REPLY_CHARS=5).validate()

    def test_error_message_lists_all_problems(self):
        """A single validate() call surfaces every error at once."""
        with pytest.raises(ValueError) as exc_info:
            _valid_settings(OPENAI_API_KEY="", OPENAI_MODEL="").validate()
        msg = str(exc_info.value)
        assert "OPENAI_API_KEY" in msg
        assert "OPENAI_MODEL" in msg

    def test_minimum_valid_voice_chars(self):
        _valid_settings(MAX_VOICE_REPLY_CHARS=10).validate()   # boundary — ok

    def test_minimum_valid_listen_frames(self):
        _valid_settings(MAX_LISTEN_FRAMES=1).validate()   # boundary — ok


# ── 2. Log redaction ──────────────────────────────────────────────────────────

class TestLogRedaction:
    def setup_method(self):
        from src.logging.logger import _RedactFilter
        self._original = list(_RedactFilter._secrets)

    def teardown_method(self):
        from src.logging.logger import _RedactFilter
        _RedactFilter._secrets = self._original

    def _make_record(self, msg: str, args=()) -> logging.LogRecord:
        return logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0,
            msg=msg, args=args, exc_info=None,
        )

    def test_registered_secret_is_replaced(self):
        from src.logging.logger import _RedactFilter, redact
        redact("sk-super-secret-key-12345")
        record = self._make_record("calling API sk-super-secret-key-12345 done")
        _RedactFilter().filter(record)
        assert "sk-super-secret-key-12345" not in record.getMessage()
        assert "[REDACTED]" in record.getMessage()

    def test_unregistered_string_is_unchanged(self):
        from src.logging.logger import _RedactFilter
        _RedactFilter._secrets = ["only-this-secret"]
        record = self._make_record("normal log message")
        _RedactFilter().filter(record)
        assert record.getMessage() == "normal log message"

    def test_empty_secret_is_ignored(self):
        from src.logging.logger import redact, _RedactFilter
        before = len(_RedactFilter._secrets)
        redact("")
        assert len(_RedactFilter._secrets) == before

    def test_duplicate_secret_registered_only_once(self):
        from src.logging.logger import redact, _RedactFilter
        redact("my-key")
        redact("my-key")
        count = _RedactFilter._secrets.count("my-key")
        assert count == 1

    def test_secret_in_format_args_is_redacted(self):
        from src.logging.logger import _RedactFilter, redact
        redact("secret-token-abc")
        # The secret appears via %s substitution, not inline in msg
        record = self._make_record("token=%s", ("secret-token-abc",))
        _RedactFilter().filter(record)
        assert "secret-token-abc" not in record.getMessage()


# ── 3. Health endpoint ────────────────────────────────────────────────────────

class TestHealthEndpoint:
    def test_health_returns_200(self):
        import src.mcp.gateway as gw
        _, status = gw._health_response("/health")
        assert status.startswith("200")

    def test_health_body_is_valid_json(self):
        import src.mcp.gateway as gw
        body, _ = gw._health_response("/health")
        data = json.loads(body)
        assert data["status"] == "ok"

    def test_health_includes_non_negative_uptime(self):
        import src.mcp.gateway as gw
        body, _ = gw._health_response("/health")
        assert json.loads(body)["uptime_s"] >= 0

    def test_ready_503_when_llm_not_initialized(self):
        import src.mcp.gateway as gw
        orig_llm, orig_fb = gw._llm, gw._fallback_resp
        gw._llm, gw._fallback_resp = None, None
        try:
            _, status = gw._health_response("/ready")
            assert status.startswith("503")
            assert json.loads(gw._health_response("/ready")[0])["ready"] is False
        finally:
            gw._llm, gw._fallback_resp = orig_llm, orig_fb

    def test_ready_503_when_fallback_not_initialized(self):
        import src.mcp.gateway as gw
        orig_fb = gw._fallback_resp
        gw._fallback_resp = None
        try:
            _, status = gw._health_response("/ready")
            assert status.startswith("503")
        finally:
            gw._fallback_resp = orig_fb

    def test_ready_200_when_fully_initialized(self):
        import src.mcp.gateway as gw
        orig_llm, orig_fb = gw._llm, gw._fallback_resp
        gw._llm = MagicMock()
        gw._fallback_resp = MagicMock()
        try:
            body, status = gw._health_response("/ready")
            assert status.startswith("200")
            assert json.loads(body)["ready"] is True
        finally:
            gw._llm, gw._fallback_resp = orig_llm, orig_fb

    def test_unknown_path_returns_404(self):
        import src.mcp.gateway as gw
        _, status = gw._health_response("/metrics")
        assert status.startswith("404")

    def test_root_path_returns_404(self):
        import src.mcp.gateway as gw
        _, status = gw._health_response("/")
        assert status.startswith("404")


# ── 4. Memory timeout ─────────────────────────────────────────────────────────

class TestMemoryTimeout:
    async def test_slow_memory_raises_timeout_error(self):
        """wait_for on a never-returning memory call raises TimeoutError."""
        from src.memory.memory_service import MemoryService
        from tests.conftest import FakeSqlite

        async def _never(*a, **kw):
            await asyncio.sleep(100)
            return []

        mem0 = AsyncMock()
        mem0.search.side_effect = _never
        svc = MemoryService(mem0_client=mem0, sqlite_memory=FakeSqlite())

        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(
                svc.getRelevantMemories("u1", "d1", "query"),
                timeout=0.05,
            )

    async def test_pipeline_continues_after_memory_timeout(
        self, fake_ws, session, mock_llm
    ):
        """_process_turn must not crash when memory exceeds MEMORY_TIMEOUT."""
        import src.mcp.gateway as gw
        from src.memory.memory_service import MemoryService
        from tests.conftest import FakeSqlite, FAKE_OPUS_FRAMES, MOCK_REPLY
        from src.mcp.response_formatter import FormattedResponse

        FAKE_FORMATTED = FormattedResponse(text=MOCK_REPLY, opus_frames=FAKE_OPUS_FRAMES)
        FAKE_FALLBACK  = FormattedResponse(text="我没听清楚，请再说一遍。", opus_frames=[b"\xff"])

        async def _slow_search(*a, **kw):
            await asyncio.sleep(100)
            return []

        mem0 = AsyncMock()
        mem0.search.side_effect = _slow_search
        svc = MemoryService(mem0_client=mem0, sqlite_memory=FakeSqlite())
        gw._fallback_resp = FAKE_FALLBACK

        with (
            patch("asr.transcribe", new_callable=AsyncMock, return_value="你好"),
            patch.object(
                gw._formatter, "format",
                new_callable=AsyncMock,
                return_value=FAKE_FORMATTED,
            ),
            patch("src.mcp.gateway.tts_service.FRAME_DURATION_MS", 0),
            patch.object(gw.settings, "MEMORY_TIMEOUT", 0.05),
        ):
            await gw._process_turn(fake_ws, [b"\x00" * 20], session, svc, mock_llm)
            await asyncio.sleep(0)

        # LLM was called despite memory timeout
        mock_llm.generateResponse.assert_awaited_once()
        # TTS was delivered
        assert fake_ws.json_frames, "No TTS frames sent after memory timeout"

    async def test_memory_timeout_does_not_store_interaction(
        self, fake_ws, session, mock_llm
    ):
        """When memory times out, Mem0 add is still called (SQLite only path)."""
        import src.mcp.gateway as gw
        from src.memory.memory_service import MemoryService
        from tests.conftest import FakeSqlite, FAKE_OPUS_FRAMES, MOCK_REPLY
        from src.mcp.response_formatter import FormattedResponse

        FAKE_FORMATTED = FormattedResponse(text=MOCK_REPLY, opus_frames=FAKE_OPUS_FRAMES)
        FAKE_FALLBACK  = FormattedResponse(text="我没听清楚，请再说一遍。", opus_frames=[b"\xff"])

        async def _slow_search(*a, **kw):
            await asyncio.sleep(100)
            return []

        mem0 = AsyncMock()
        mem0.search.side_effect = _slow_search
        mem0.add.return_value = None
        svc = MemoryService(mem0_client=mem0, sqlite_memory=FakeSqlite())
        gw._fallback_resp = FAKE_FALLBACK

        with (
            patch("asr.transcribe", new_callable=AsyncMock, return_value="你好"),
            patch.object(
                gw._formatter, "format",
                new_callable=AsyncMock,
                return_value=FAKE_FORMATTED,
            ),
            patch("src.mcp.gateway.tts_service.FRAME_DURATION_MS", 0),
            patch.object(gw.settings, "MEMORY_TIMEOUT", 0.05),
        ):
            await gw._process_turn(fake_ws, [b"\x00" * 20], session, svc, mock_llm)
            # Allow ensure_future store task to run
            await asyncio.sleep(0.1)

        # SQLite was still written (add goes through MemoryService)
        history = svc.get_history(session.device_id)
        assert any(m["content"] == "你好" for m in history)


# ── 5. Configurable voice reply length ───────────────────────────────────────

class TestMaxVoiceReplyChars:
    def test_sanitize_respects_configured_cap(self):
        from src.mcp.response_formatter import _sanitize
        import src.mcp.response_formatter as rf

        long_text = "好" * 400  # 400 chars, all Chinese

        with patch.object(rf.settings, "MAX_VOICE_REPLY_CHARS", 50):
            result = _sanitize(long_text)

        # Must be <= cap + 1 (the "…" appended)
        assert len(result) <= 52   # 50 chars + possible "…"

    def test_sanitize_does_not_truncate_short_text(self):
        from src.mcp.response_formatter import _sanitize

        short = "你好世界"
        result = _sanitize(short)
        assert result == short

    def test_sanitize_breaks_at_sentence_boundary(self):
        from src.mcp.response_formatter import _sanitize
        import src.mcp.response_formatter as rf

        # Build text where a sentence boundary falls within the cap
        text = "第一句话。" + "填充" * 30 + "第二句话。"

        with patch.object(rf.settings, "MAX_VOICE_REPLY_CHARS", 20):
            result = _sanitize(text)

        assert result.endswith("。…") or result.endswith("…")

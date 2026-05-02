"""
OpenAIClient conversation threading via previous_response_id.

Tests verify:
  - First turn: no previous_response_id, response_id stored
  - Second turn: previous_response_id sent, usage["threaded"] == True
  - Stale/expired ID: cleared on error, full history retried
  - user_id="" : threading disabled, no ID stored
  - clear_thread(): removes stored ID
  - Gateway passes session.user_id to generateResponse
"""
import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from src.ai.openai_client import OpenAIClient, FALLBACK_RESPONSE, DEFAULT_TIMEOUT


# ── Helpers ───────────────────────────────────────────────────────────────────

USER_MSG = {"role": "user", "content": "Hello"}
HISTORY   = [
    {"role": "user",      "content": "First message"},
    {"role": "assistant", "content": "First reply"},
    {"role": "user",      "content": "Hello"},
]

def _make_resp(response_id: str, text: str = "Hi") -> MagicMock:
    r = MagicMock()
    r.output_text = text
    r.id = response_id
    r.model = "gpt-4o-mini"
    r.usage = MagicMock(input_tokens=10, output_tokens=5, total_tokens=15)
    return r


def _make_client() -> tuple[OpenAIClient, AsyncMock]:
    client = OpenAIClient.__new__(OpenAIClient)
    client.model = "gpt-4o-mini"
    client._prev_ids = {}
    mock_create = AsyncMock()
    client._client = MagicMock()
    client._client.responses = MagicMock()
    client._client.responses.create = mock_create
    return client, mock_create


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestFirstTurn:
    async def test_no_previous_response_id_on_first_call(self):
        client, create = _make_client()
        create.return_value = _make_resp("resp_001")

        await client.generateResponse([USER_MSG], user_id="u1")

        call_kwargs = create.call_args.kwargs
        assert "previous_response_id" not in call_kwargs

    async def test_response_id_stored_after_first_call(self):
        client, create = _make_client()
        create.return_value = _make_resp("resp_001")

        await client.generateResponse([USER_MSG], user_id="u1")

        assert client._prev_ids["u1"] == "resp_001"

    async def test_full_history_sent_on_first_call(self):
        client, create = _make_client()
        create.return_value = _make_resp("resp_001")

        await client.generateResponse(HISTORY, user_id="u1")

        call_kwargs = create.call_args.kwargs
        assert call_kwargs["input"] == HISTORY


class TestSecondTurn:
    async def test_previous_response_id_sent_on_second_call(self):
        client, create = _make_client()
        create.side_effect = [
            _make_resp("resp_001"),
            _make_resp("resp_002"),
        ]

        await client.generateResponse([USER_MSG], user_id="u1")
        await client.generateResponse(HISTORY, user_id="u1")

        second_kwargs = create.call_args.kwargs
        assert second_kwargs.get("previous_response_id") == "resp_001"

    async def test_only_last_user_message_sent_on_second_call(self):
        client, create = _make_client()
        create.side_effect = [
            _make_resp("resp_001"),
            _make_resp("resp_002"),
        ]

        await client.generateResponse([USER_MSG], user_id="u1")
        await client.generateResponse(HISTORY, user_id="u1")

        second_kwargs = create.call_args.kwargs
        # Only the last user message — not the full 3-message history
        assert len(second_kwargs["input"]) == 1
        assert second_kwargs["input"][0]["role"] == "user"
        assert second_kwargs["input"][0]["content"] == "Hello"

    async def test_usage_threaded_true_on_second_call(self):
        client, create = _make_client()
        create.side_effect = [
            _make_resp("resp_001"),
            _make_resp("resp_002"),
        ]

        await client.generateResponse([USER_MSG], user_id="u1")
        _, usage = await client.generateResponse(HISTORY, user_id="u1")

        assert usage.get("threaded") is True

    async def test_id_updated_after_second_call(self):
        client, create = _make_client()
        create.side_effect = [
            _make_resp("resp_001"),
            _make_resp("resp_002"),
        ]

        await client.generateResponse([USER_MSG], user_id="u1")
        await client.generateResponse(HISTORY, user_id="u1")

        assert client._prev_ids["u1"] == "resp_002"


class TestStaleIdFallback:
    async def test_stale_id_cleared_on_api_error(self):
        from openai import APIStatusError
        import httpx

        client, create = _make_client()

        # First call: store an ID
        create.return_value = _make_resp("resp_stale")
        await client.generateResponse([USER_MSG], user_id="u1")
        assert "u1" in client._prev_ids

        # Second call: threaded path raises 404 (expired ID), full history succeeds
        expired_exc = APIStatusError(
            "Not found",
            response=MagicMock(status_code=404),
            body={"error": {"message": "Not found"}},
        )
        create.side_effect = [expired_exc, _make_resp("resp_new")]
        await client.generateResponse(HISTORY, user_id="u1")

        # Stale ID is gone; new ID stored
        assert client._prev_ids.get("u1") == "resp_new"

    async def test_fallback_to_full_history_on_stale_id(self):
        from openai import APIStatusError

        client, create = _make_client()

        create.return_value = _make_resp("resp_stale")
        await client.generateResponse([USER_MSG], user_id="u1")

        expired_exc = APIStatusError(
            "Gone",
            response=MagicMock(status_code=410),
            body={"error": {"message": "Gone"}},
        )
        create.side_effect = [expired_exc, _make_resp("resp_new")]
        await client.generateResponse(HISTORY, user_id="u1")

        # Second create call receives the full history, not just one message
        second_call_kwargs = create.call_args.kwargs
        assert second_call_kwargs["input"] == HISTORY
        assert "previous_response_id" not in second_call_kwargs


class TestThreadingDisabled:
    async def test_empty_user_id_disables_threading(self):
        client, create = _make_client()
        create.side_effect = [
            _make_resp("resp_001"),
            _make_resp("resp_002"),
        ]

        await client.generateResponse([USER_MSG], user_id="")
        _, usage = await client.generateResponse(HISTORY, user_id="")

        # No ID stored, no previous_response_id in either call, no threaded flag
        assert "" not in client._prev_ids
        assert "threaded" not in usage
        for call in create.call_args_list:
            assert "previous_response_id" not in call.kwargs

    async def test_no_user_id_arg_disables_threading(self):
        client, create = _make_client()
        create.side_effect = [
            _make_resp("resp_001"),
            _make_resp("resp_002"),
        ]

        # user_id defaults to ""
        await client.generateResponse([USER_MSG])
        await client.generateResponse(HISTORY)

        assert client._prev_ids == {}


class TestClearThread:
    async def test_clear_thread_removes_stored_id(self):
        client, create = _make_client()
        create.return_value = _make_resp("resp_001")

        await client.generateResponse([USER_MSG], user_id="u1")
        assert "u1" in client._prev_ids

        client.clear_thread("u1")
        assert "u1" not in client._prev_ids

    async def test_clear_thread_noop_for_unknown_user(self):
        client, _ = _make_client()
        client.clear_thread("nonexistent")   # must not raise

    async def test_after_clear_next_call_is_first_turn(self):
        client, create = _make_client()
        create.side_effect = [
            _make_resp("resp_001"),
            _make_resp("resp_002"),
        ]

        await client.generateResponse([USER_MSG], user_id="u1")
        client.clear_thread("u1")
        await client.generateResponse(HISTORY, user_id="u1")

        # Second call should behave like a first turn — no previous_response_id
        second_kwargs = create.call_args.kwargs
        assert "previous_response_id" not in second_kwargs
        assert second_kwargs["input"] == HISTORY


class TestGatewayPassesUserId:
    """Verify the gateway forwards session.user_id to generateResponse."""

    async def test_user_id_passed_to_generate_response(
        self, fake_ws, session, mock_llm
    ):
        import src.mcp.gateway as gw
        from src.memory.memory_service import MemoryService
        from tests.conftest import FakeSqlite, FAKE_OPUS_FRAMES, MOCK_REPLY
        from src.mcp.response_formatter import FormattedResponse
        from unittest.mock import patch, AsyncMock

        FAKE_FORMATTED = FormattedResponse(text=MOCK_REPLY, opus_frames=FAKE_OPUS_FRAMES)
        FAKE_FALLBACK  = FormattedResponse(text="我没听清楚，请再说一遍。", opus_frames=[b"\xff\xff"])

        mem0 = AsyncMock()
        mem0.search.return_value = []
        mem0.add.return_value = None
        svc = MemoryService(mem0_client=mem0, sqlite_memory=FakeSqlite())

        captured: dict = {}

        async def capturing_generate(input_messages, context="", user_id="", **kw):
            captured["user_id"] = user_id
            return MOCK_REPLY, {"model": "test", "total_tokens": 10, "response_id": "r1"}

        mock_llm.generateResponse = capturing_generate
        gw._fallback_resp = FAKE_FALLBACK

        with (
            patch("asr.transcribe", new_callable=AsyncMock, return_value="你好"),
            patch.object(
                gw._formatter, "format",
                new_callable=AsyncMock,
                return_value=FAKE_FORMATTED,
            ),
            patch("src.mcp.gateway.tts_service.FRAME_DURATION_MS", 0),
        ):
            await gw._process_turn(fake_ws, [b"\x00" * 20], session, svc, mock_llm)
            await __import__("asyncio").sleep(0)

        assert "user_id" in captured, "generateResponse was never called"
        assert captured["user_id"] == session.user_id, (
            f"Expected user_id={session.user_id!r}, got {captured['user_id']!r}"
        )

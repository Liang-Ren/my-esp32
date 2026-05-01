"""
Test 1 — ESP32 message parsing (ProtocolAdapter)
Test 7 — Response format remains ESP32-compatible

These tests are synchronous and need no mocks beyond the adapter itself.
"""
import json
import struct
import pytest

from src.mcp.protocol_adapter import ProtocolAdapter


@pytest.fixture(scope="module")
def adapter() -> ProtocolAdapter:
    return ProtocolAdapter()


# ── Test 1: Hello message parsing ─────────────────────────────────────────────

class TestHelloParsing:
    def test_parse_v1(self, adapter):
        msg = {"type": "hello", "session_id": "abc123", "version": 1}
        hello = adapter.parse_hello(msg)
        assert hello.session_id == "abc123"
        assert hello.version == 1

    def test_parse_v2(self, adapter):
        msg = {"type": "hello", "session_id": "xyz789", "version": 2}
        hello = adapter.parse_hello(msg)
        assert hello.version == 2

    def test_parse_v3(self, adapter):
        msg = {"type": "hello", "session_id": "xyz789", "version": 3}
        hello = adapter.parse_hello(msg)
        assert hello.version == 3

    def test_missing_version_defaults_to_1(self, adapter):
        hello = adapter.parse_hello({"type": "hello", "session_id": "abc"})
        assert hello.version == 1

    def test_missing_session_id_is_empty_string(self, adapter):
        hello = adapter.parse_hello({"type": "hello", "version": 1})
        assert hello.session_id == ""

    def test_session_id_preserved_exactly(self, adapter):
        sid = "session-uuid-1234-5678-abcd"
        hello = adapter.parse_hello({"type": "hello", "session_id": sid, "version": 1})
        assert hello.session_id == sid


# ── Test 1: Listen message parsing ────────────────────────────────────────────

class TestListenParsing:
    def test_start_auto(self, adapter):
        listen = adapter.parse_listen({"type": "listen", "state": "start", "mode": "auto"})
        assert listen.state == "start"
        assert listen.mode == "auto"

    def test_start_realtime(self, adapter):
        listen = adapter.parse_listen({"type": "listen", "state": "start", "mode": "realtime"})
        assert listen.state == "start"
        assert listen.mode == "realtime"

    def test_stop(self, adapter):
        listen = adapter.parse_listen({"type": "listen", "state": "stop"})
        assert listen.state == "stop"
        assert listen.mode == "auto"   # default when missing

    def test_detect(self, adapter):
        listen = adapter.parse_listen({"type": "listen", "state": "detect", "mode": "manual"})
        assert listen.state == "detect"
        assert listen.mode == "manual"

    def test_missing_state_is_empty(self, adapter):
        listen = adapter.parse_listen({"type": "listen"})
        assert listen.state == ""


# ── Test 7: Outbound message format ───────────────────────────────────────────

class TestResponseFormat:
    def test_hello_response_schema(self, adapter):
        raw = adapter.build_hello("session-001")
        data = json.loads(raw)

        assert data["type"] == "hello"
        assert data["transport"] == "websocket"
        assert data["session_id"] == "session-001"
        assert "audio_params" in data
        assert data["audio_params"]["format"] == "opus"
        assert isinstance(data["audio_params"]["sample_rate"], int)
        assert isinstance(data["audio_params"]["channels"], int)
        assert isinstance(data["audio_params"]["frame_duration"], int)

    def test_tts_start_schema(self, adapter):
        raw = adapter.build_tts_start("session-001")
        data = json.loads(raw)
        assert data["type"] == "tts"
        assert data["state"] == "start"
        assert data["session_id"] == "session-001"

    def test_tts_sentence_schema(self, adapter):
        raw = adapter.build_tts_sentence("你好！", "session-001")
        data = json.loads(raw)
        assert data["type"] == "tts"
        assert data["state"] == "sentence_start"
        assert data["text"] == "你好！"
        assert data["session_id"] == "session-001"

    def test_tts_stop_schema(self, adapter):
        raw = adapter.build_tts_stop("session-001")
        data = json.loads(raw)
        assert data["type"] == "tts"
        assert data["state"] == "stop"
        assert data["session_id"] == "session-001"

    def test_tts_sequence_uses_same_session_id(self, adapter):
        sid = "session-xyz"
        start = json.loads(adapter.build_tts_start(sid))
        sentence = json.loads(adapter.build_tts_sentence("hi", sid))
        stop = json.loads(adapter.build_tts_stop(sid))
        assert start["session_id"] == sid
        assert sentence["session_id"] == sid
        assert stop["session_id"] == sid

    def test_opus_frame_v1_is_raw(self, adapter):
        opus = b"\xaa\xbb\xcc\xdd"
        frame = adapter.wrap_opus_frame(opus, version=1)
        assert frame == opus

    def test_opus_frame_v2_has_14_byte_header(self, adapter):
        opus = b"\xaa\xbb\xcc"
        frame = adapter.wrap_opus_frame(opus, version=2)
        assert len(frame) == 14 + len(opus)
        version_field, _, _, _, length = struct.unpack(">HHHII", frame[:14])
        assert version_field == 2
        assert length == len(opus)
        assert frame[14:] == opus

    def test_opus_frame_v3_has_4_byte_header(self, adapter):
        opus = b"\xaa\xbb\xcc"
        frame = adapter.wrap_opus_frame(opus, version=3)
        assert len(frame) == 4 + len(opus)
        assert frame[:2] == b"\x00\x00"
        (length,) = struct.unpack(">H", frame[2:4])
        assert length == len(opus)
        assert frame[4:] == opus

    def test_all_tts_messages_are_json_strings(self, adapter):
        sid = "s1"
        for raw in [
            adapter.build_tts_start(sid),
            adapter.build_tts_sentence("text", sid),
            adapter.build_tts_stop(sid),
        ]:
            assert isinstance(raw, str)
            json.loads(raw)   # must not raise

    def test_opus_frames_are_bytes(self, adapter):
        for version in (1, 2, 3):
            frame = adapter.wrap_opus_frame(b"\x01\x02", version=version)
            assert isinstance(frame, bytes)

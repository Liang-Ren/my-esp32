"""
WebSocket gateway — wires the full ESP32 ↔ AI request flow.

Per-request pipeline (each step is timed and logged):
    audio frames received
      -> [asr]     faster-whisper transcribes Opus → text
      -> [parse]   mode detection + history fetch
      -> [memory]  Mem0 semantic search + user profile (parallel, with timeout)
      -> [prompt]  build_input assembles instructions + messages
      -> [llm]     OpenAI Responses API returns reply
      -> [format]  ResponseFormatter sanitizes text + generates Opus frames
      -> [store]   addInteraction persists to Mem0 + SQLite (non-blocking)
      -> [send]    TTS JSON + binary frames streamed to ESP32

Error isolation:
    - Mem0 down / slow → continue with empty memories (logged as WARNING)
    - Memory timeout   → continue with empty context (logged as WARNING)
    - OpenAI down      → send FALLBACK_TEXT TTS, no interaction stored
    - Malformed frame  → log and skip (ESP32 connection preserved)

Health server (HTTP on HEALTH_PORT):
    GET /health  → 200 {"status":"ok","uptime_s":N}
    GET /ready   → 200/503 {"ready":true|false}

WebSocket heartbeat:
    Automated ping/pong via websockets library (WS_PING_INTERVAL / WS_PING_TIMEOUT).
    Unresponsive devices are disconnected automatically.
"""
import sys
import asyncio
import json
import time
from pathlib import Path

_ROOT = Path(__file__).parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import websockets

from src.config.settings import settings
from src.logging.logger import log, log_step, log_request, new_request_id
from src.mcp.protocol_adapter import ProtocolAdapter
from src.mcp.session_manager import SessionManager, Session
from src.mcp.response_formatter import ResponseFormatter, FormattedResponse
from src.memory.memory_service import MemoryService
from src.ai.openai_client import OpenAIClient, FALLBACK_RESPONSE
from src.ai.model_router import ModelRouter
from src.ai.prompt_builder import build_input

import asr
import tts_service

FALLBACK_TEXT = "我没听清楚，请再说一遍。"
_fallback_resp: FormattedResponse | None = None   # pre-built at startup

_adapter = ProtocolAdapter()
_session_mgr = SessionManager(settings.MEM0_USER_ID_PREFIX)
_router = ModelRouter()
_formatter = ResponseFormatter()
_llm: OpenAIClient | None = None   # singleton — _prev_ids survive reconnects

_start_time: float = time.time()
_device_last_seen: dict[str, float] = {}   # device_id → disconnect timestamp


# ── Helpers ────────────────────────────────────────────────────────────────────

def _ms(t0: float) -> int:
    return round((time.time() - t0) * 1000)


async def _send_tts(ws, resp: FormattedResponse, session: Session) -> None:
    """Stream a FormattedResponse to the ESP32."""
    sid = session.session_id
    ver = session.proto_version
    await ws.send(_adapter.build_tts_start(sid))
    await ws.send(_adapter.build_tts_sentence(resp.text, sid))
    for frame in resp.opus_frames:
        await ws.send(_adapter.wrap_opus_frame(frame, ver))
        await asyncio.sleep(tts_service.FRAME_DURATION_MS / 1000)
    await ws.send(_adapter.build_tts_stop(sid))


async def _init_fallback() -> None:
    global _fallback_resp
    frames = await tts_service.generate(FALLBACK_TEXT)
    _fallback_resp = FormattedResponse(text=FALLBACK_TEXT, opus_frames=frames)
    log(f"Fallback TTS ready ({len(frames)} frames)")


# ── Health server ──────────────────────────────────────────────────────────────

def _health_response(path: str) -> tuple[bytes, str]:
    """Return (body_bytes, http_status_line) for a health-check path."""
    if path == "/health":
        body = json.dumps({
            "status": "ok",
            "uptime_s": round(time.time() - _start_time),
        }).encode()
        return body, "200 OK"
    if path == "/ready":
        ready = _llm is not None and _fallback_resp is not None
        body = json.dumps({"ready": ready}).encode()
        return body, ("200 OK" if ready else "503 Service Unavailable")
    return b'{"error":"not found"}', "404 Not Found"


async def _health_server(host: str, port: int) -> None:
    async def _handle(
        reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        try:
            data = await asyncio.wait_for(reader.read(2048), timeout=2.0)
            first_line = data.decode(errors="ignore").split("\r\n")[0]
            parts = first_line.split()
            path = parts[1] if len(parts) >= 2 else "/"
            body, status = _health_response(path)
            response = (
                f"HTTP/1.1 {status}\r\n"
                f"Content-Type: application/json\r\n"
                f"Content-Length: {len(body)}\r\n"
                f"Connection: close\r\n\r\n"
            ).encode() + body
            writer.write(response)
            await writer.drain()
        except Exception:
            pass
        finally:
            writer.close()

    srv = await asyncio.start_server(_handle, host, port)
    log(f"Health on {host}:{port}")
    async with srv:
        await srv.serve_forever()


# ── Core request pipeline ──────────────────────────────────────────────────────

async def _process_turn(
    ws,
    audio_frames: list[bytes],
    session: Session,
    memory_svc: MemoryService,
    llm: OpenAIClient,
) -> None:
    request_id = new_request_id()
    t_total = time.time()
    metrics: dict = {}
    user_text = ""
    response_text = FALLBACK_TEXT
    model = "?"
    error: str | None = None

    try:
        # ── Step 1: ASR ────────────────────────────────────────────────────────
        t = time.time()
        log_step(request_id, "asr", f"decoding {len(audio_frames)} Opus frame(s)…")
        user_text = await asr.transcribe(audio_frames)
        metrics["asr_ms"] = _ms(t)
        log_step(request_id, "asr", f"→ {user_text!r}", ms=metrics["asr_ms"])

        if not user_text:
            log_step(request_id, "asr", "empty — sending fallback")
            await _send_tts(ws, _fallback_resp, session)
            metrics["total_ms"] = _ms(t_total)
            log_request(request_id, session.device_id, "", FALLBACK_TEXT, "?", metrics)
            return

        # ── Step 2: Parse ──────────────────────────────────────────────────────
        t = time.time()
        mode = _router.detect_mode(user_text)
        history = memory_svc.get_history(session.device_id)
        metrics["parse_ms"] = _ms(t)
        log_step(request_id, "parse",
                 f"mode={mode} history={len(history)}turns",
                 ms=metrics["parse_ms"])

        # ── Step 3: Memory (with timeout) ──────────────────────────────────────
        t = time.time()
        log_step(request_id, "memory", "retrieving…")
        try:
            memories, profile = await asyncio.wait_for(
                asyncio.gather(
                    memory_svc.getRelevantMemories(
                        session.user_id, session.device_id, user_text
                    ),
                    memory_svc.getUserProfile(session.user_id, session.device_id),
                ),
                timeout=settings.MEMORY_TIMEOUT,
            )
            metrics["memory_ms"] = _ms(t)
            log_step(request_id, "memory",
                     f"{len(memories)} relevant, summary={bool(profile['summary'])}",
                     ms=metrics["memory_ms"])
        except asyncio.TimeoutError:
            memories = []
            profile = {
                "summary": "", "facts": [], "preferences": {}, "preferences_str": ""
            }
            metrics["memory_ms"] = _ms(t)
            log_step(request_id, "memory",
                     "TIMEOUT — continuing without memory context",
                     ms=metrics["memory_ms"])

        # ── Step 4: Prompt ─────────────────────────────────────────────────────
        instructions, input_msgs = build_input(
            user_text, history,
            memory_summary=profile["summary"],
            recent_memory=memories,
            user_preferences=profile["preferences_str"],
            mode=mode,
        )
        log_step(request_id, "prompt",
                 f"instructions={len(instructions)}chars "
                 f"messages={len(input_msgs)}")

        # ── Step 5: LLM ────────────────────────────────────────────────────────
        t = time.time()
        log_step(request_id, "llm", f"calling {llm.model}…")
        response_text, usage = await llm.generateResponse(
            input_msgs, context=instructions, user_id=session.user_id,
            timeout=settings.LLM_TIMEOUT,
        )
        metrics["llm_ms"] = _ms(t)
        model = usage.get("model", llm.model)

        if usage.get("error"):
            response_text = FALLBACK_TEXT
            error = usage["error"]
            log_step(request_id, "llm", f"ERROR: {error}", ms=metrics["llm_ms"])
            await _send_tts(ws, _fallback_resp, session)
            metrics["total_ms"] = _ms(t_total)
            log_request(request_id, session.device_id, user_text, response_text,
                        model, metrics, error=error)
            return

        log_step(request_id, "llm",
                 f"→ {response_text[:60]!r}…"
                 if len(response_text) > 60 else f"→ {response_text!r}",
                 ms=metrics["llm_ms"])
        metrics["tokens"] = usage.get("total_tokens", 0)

        # ── Step 6: Format ─────────────────────────────────────────────────────
        t = time.time()
        formatted = await _formatter.format(response_text)
        metrics["tts_ms"] = _ms(t)
        response_text = formatted.text
        log_step(request_id, "format",
                 f"{len(formatted.opus_frames)} Opus frames",
                 ms=metrics["tts_ms"])

        # ── Step 7: Store (non-blocking) ───────────────────────────────────────
        log_step(request_id, "store", "queued")
        asyncio.ensure_future(
            memory_svc.addInteraction(
                session.user_id, session.device_id, user_text, formatted.text
            )
        )

        # ── Step 8: Send ───────────────────────────────────────────────────────
        log_step(request_id, "send", f"streaming {len(formatted.opus_frames)} frames…")
        await _send_tts(ws, formatted, session)
        log_step(request_id, "send", "done")

    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        log_step(request_id, "ERROR", error)
        try:
            await _send_tts(ws, _fallback_resp, session)
        except Exception:
            pass

    metrics["total_ms"] = _ms(t_total)
    log_request(
        request_id, session.device_id, user_text, response_text,
        model, metrics, error=error,
    )


# ── Service wiring ─────────────────────────────────────────────────────────────

def _build_memory_service() -> MemoryService:
    from src.memory.mem0_client import Mem0Client, OpenMemoryClient, MockMem0Client
    from memory import Memory as SqliteMemory

    if settings.MEM0_API_KEY:
        log("Mem0: cloud mode")
        mem0 = Mem0Client(settings.MEM0_API_KEY)
    elif settings.MEM0_SERVER_URL:
        log(f"Mem0: self-hosted OpenMemory at {settings.MEM0_SERVER_URL}")
        mem0 = OpenMemoryClient(settings.MEM0_SERVER_URL)
    else:
        log("Mem0: not configured — using in-process mock")
        mem0 = MockMem0Client()

    return MemoryService(mem0_client=mem0, sqlite_memory=SqliteMemory())


def _init_llm() -> None:
    global _llm
    _llm = OpenAIClient(settings.OPENAI_API_KEY, settings.OPENAI_MODEL)
    log(f"LLM: {settings.OPENAI_MODEL} (timeout={settings.LLM_TIMEOUT}s)")


# ── WebSocket handler ──────────────────────────────────────────────────────────

async def handler(ws) -> None:
    addr = ws.remote_address
    log(f"[+] {addr}")

    session: Session | None = None
    audio_frames: list[bytes] = []
    audio_frame_count = 0
    responding = False
    silence_task = None
    # Realtime-mode VAD state: Opus speech frames are much larger than silence frames
    rt_speech_seen = False
    rt_silence_count = 0

    memory_svc = _build_memory_service()
    llm = _llm

    async def on_silence() -> None:
        nonlocal responding, audio_frames, audio_frame_count
        await asyncio.sleep(settings.SILENCE_TIMEOUT)
        if audio_frame_count > 0 and not responding:
            log(f"  [{session.device_id}] silence → {audio_frame_count} frames")
            responding = True
            frames, audio_frames, audio_frame_count = audio_frames.copy(), [], 0
            await _process_turn(ws, frames, session, memory_svc, llm)
            responding = False

    def reset_silence_timer() -> None:
        nonlocal silence_task
        if silence_task and not silence_task.done():
            silence_task.cancel()
        silence_task = asyncio.create_task(on_silence())

    try:
        # ── Handshake ──────────────────────────────────────────────────────────
        raw = await asyncio.wait_for(ws.recv(), timeout=15)
        try:
            msg = json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            log(f"[!] {addr} malformed hello: {exc} — closing")
            return

        if msg.get("type") != "hello":
            log(f"[!] {addr} unexpected first message type={msg.get('type')!r} — closing")
            return

        hello = _adapter.parse_hello(msg)
        session = _session_mgr.create(addr, hello.session_id, hello.version)

        # Reconnect detection: same device_id seen before?
        if session.device_id in _device_last_seen:
            gap = round(time.time() - _device_last_seen[session.device_id])
            log(f"  [{session.device_id}] reconnect after {gap}s — LLM thread preserved")
        else:
            log(f"  hello v{session.proto_version} "
                f"session={session.session_id[:8]} "
                f"device={session.device_id} "
                f"user={session.user_id}")

        await ws.send(_adapter.build_hello(session.session_id))

        # ── Message loop ───────────────────────────────────────────────────────
        async for message in ws:
            if isinstance(message, bytes):
                if responding:
                    continue
                audio_frames.append(message)
                audio_frame_count += 1
                if audio_frame_count == 1:
                    log(f"  [{session.device_id}] audio start (mode={session.listen_mode})")

                if session.listen_mode == "realtime":
                    # Frame-size VAD: Opus speech frames >> silence frames in bytes.
                    # Threshold 40 bytes works for both DTX and VBR encoders.
                    _RT_SILENCE_THRESH = 40   # bytes — below this = likely silence
                    _RT_SILENCE_NEEDED = 10   # consecutive silent frames (10×60ms = 600ms)
                    if len(message) > _RT_SILENCE_THRESH:
                        rt_speech_seen = True
                        rt_silence_count = 0
                    else:
                        rt_silence_count += 1

                    early_stop = rt_speech_seen and rt_silence_count >= _RT_SILENCE_NEEDED
                    hit_max = audio_frame_count >= settings.MAX_LISTEN_FRAMES

                    if early_stop or hit_max:
                        reason = "silence after speech" if early_stop else "max frames"
                        log(f"  [{session.device_id}] {reason} → {audio_frame_count} frames")
                        rt_speech_seen = False
                        rt_silence_count = 0
                        responding = True
                        frames, audio_frames, audio_frame_count = audio_frames.copy(), [], 0
                        await _process_turn(ws, frames, session, memory_svc, llm)
                        responding = False
                else:
                    reset_silence_timer()

            else:
                try:
                    data = json.loads(message)
                except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                    log(f"  [{session.device_id if session else addr}] "
                        f"malformed control frame: {exc!r} — skipping")
                    continue

                mtype = data.get("type", "")
                state = data.get("state", "")
                log(f"  [{session.device_id}] << type={mtype} state={state} "
                    f"mode={data.get('mode', '')}")

                if mtype == "listen":
                    try:
                        listen = _adapter.parse_listen(data)
                    except Exception as exc:
                        log(f"  [{session.device_id}] bad listen frame: {exc} — skipping")
                        continue

                    if listen.state == "start":
                        session.listen_mode = listen.mode
                        audio_frames, audio_frame_count = [], 0
                        rt_speech_seen = False
                        rt_silence_count = 0
                    elif listen.state in ("stop", "detect"):
                        if silence_task and not silence_task.done():
                            silence_task.cancel()
                        if not responding and audio_frame_count > 0:
                            responding = True
                            frames, audio_frames, audio_frame_count = audio_frames.copy(), [], 0
                            await _process_turn(ws, frames, session, memory_svc, llm)
                            responding = False

    except websockets.exceptions.ConnectionClosed:
        log(f"[-] {addr} disconnected")
    except asyncio.TimeoutError:
        log(f"[-] {addr} hello timeout")
    except Exception as exc:
        log(f"[!] {addr} {type(exc).__name__}: {exc}")
    finally:
        if silence_task and not silence_task.done():
            silence_task.cancel()
        if session:
            _device_last_seen[session.device_id] = time.time()
            _session_mgr.remove(session.connection_id)


# ── Entry point ────────────────────────────────────────────────────────────────

async def run() -> None:
    _init_llm()
    await _init_fallback()
    log(f"WebSocket on {settings.WS_HOST}:{settings.WS_PORT} "
        f"(ping every {settings.WS_PING_INTERVAL}s)")

    async with websockets.serve(
        handler,
        settings.WS_HOST,
        settings.WS_PORT,
        ping_interval=settings.WS_PING_INTERVAL,
        ping_timeout=settings.WS_PING_TIMEOUT,
    ):
        await _health_server(settings.WS_HOST, settings.HEALTH_PORT)

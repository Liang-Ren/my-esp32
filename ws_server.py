#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Xiaozhi ESP32 WebSocket backend — memory + OpenAI Q&A."""

import sys
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

import asyncio
import json
import os
import time
from pathlib import Path

import websockets
from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv(Path(__file__).parent / ".env", override=True)

from logger import log, log_request, new_request_id
from memory import Memory
from prompt_builder import build_messages, detect_mode
from llm import LLMClient
import asr
import tts_service

HOST = "0.0.0.0"
PORT = 8001
MAX_LISTEN_FRAMES = 25   # ~1.5s of audio in realtime mode
SILENCE_TIMEOUT = 1.5    # seconds for auto/manual mode

# Shared singletons
memory = Memory()
llm = LLMClient()
openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

FALLBACK_TEXT = "我没听清楚，请再说一遍。"
_fallback_frames: list[bytes] = []


async def _init_fallback():
    global _fallback_frames
    _fallback_frames = await tts_service.generate(FALLBACK_TEXT)
    log(f"Fallback TTS ready ({len(_fallback_frames)} frames)")


async def send_tts(ws, frames: list[bytes], text: str, session_id: str, version: int):
    await ws.send(json.dumps({"type": "tts", "state": "start", "session_id": session_id}))
    await ws.send(json.dumps({"type": "tts", "state": "sentence_start",
                              "text": text, "session_id": session_id}))
    for frame in frames:
        await ws.send(tts_service.make_frame(frame, version))
        await asyncio.sleep(tts_service.FRAME_DURATION_MS / 1000)
    await ws.send(json.dumps({"type": "tts", "state": "stop", "session_id": session_id}))


async def _maybe_update_long_term(device_id: str, memory, llm):
    """Every 5 user messages, ask LLM to extract facts and update long-term memory."""
    count = memory.conn.execute(
        "SELECT COUNT(*) FROM messages WHERE device_id=? AND role='user'", (device_id,)
    ).fetchone()[0]
    if count % 5 != 0:
        return
    history = memory.get_recent(device_id, limit=10)
    if not history:
        return
    extract_prompt = [
        {"role": "system", "content": (
            "从对话中提取关于用户的重要信息。"
            "返回JSON格式：{\"summary\":\"一句话总结用户\",\"facts\":[\"事实1\",\"事实2\"]}"
            "如果没有新信息，返回 {\"summary\":\"\",\"facts\":[]}"
        )},
        *history,
        {"role": "user", "content": "请提取上述对话中关于我的信息。"}
    ]
    try:
        result, _ = await llm.complete(extract_prompt)
        import json as _json
        data = _json.loads(result)
        existing = memory.get_long_term(device_id)
        new_facts = list(dict.fromkeys(existing["facts"] + data.get("facts", [])))[:20]
        summary = data.get("summary") or existing["summary"]
        memory.update_long_term(device_id, summary=summary, facts=new_facts)
    except Exception:
        pass


async def process_turn(
    ws,
    audio_frames: list[bytes],
    session_id: str,
    device_id: str,
    proto_version: int,
):
    """ASR → memory → LLM → TTS → send."""
    request_id = new_request_id()
    t0 = time.time()
    user_text = ""
    response_text = FALLBACK_TEXT
    usage = {}

    try:
        # 1. ASR
        log(f"  [{request_id}] ASR ({len(audio_frames)} frames)…")
        user_text = await asr.transcribe(audio_frames, openai_client)
        log(f"  [{request_id}] User: {user_text!r}")

        if not user_text:
            frames = _fallback_frames
        else:
            # 2. Prompt
            history = memory.get_recent(device_id)
            long_term = memory.get_long_term(device_id)
            mode = detect_mode(user_text)
            messages = build_messages(user_text, history, long_term, mode)

            # 3. LLM
            log(f"  [{request_id}] LLM (mode={mode})…")
            response_text, usage = await llm.complete(messages)
            log(f"  [{request_id}] AI: {response_text!r}")

            # 4. Save memory
            memory.add_message(device_id, "user", user_text)
            memory.add_message(device_id, "assistant", response_text)
            asyncio.ensure_future(_maybe_update_long_term(device_id, memory, llm))

            # 5. TTS
            frames = await tts_service.generate(response_text)

        await send_tts(ws, frames, response_text, session_id, proto_version)

    except Exception as e:
        log(f"  [{request_id}] Error: {e}")
        await send_tts(ws, _fallback_frames, FALLBACK_TEXT, session_id, proto_version)
        usage = {"error": str(e)}

    latency = (time.time() - t0) * 1000
    log_request(request_id, device_id, user_text, response_text,
                usage.get("model", "?"), latency, usage,
                error=usage.get("error"))


async def handler(ws):
    addr = ws.remote_address
    log(f"[+] Connected: {addr}")

    session_id = ""
    proto_version = 1
    device_id = ""
    listen_mode = "auto"
    audio_frames: list[bytes] = []
    audio_frame_count = 0
    responding = False
    silence_task = None

    async def on_silence():
        nonlocal responding, audio_frames, audio_frame_count
        await asyncio.sleep(SILENCE_TIMEOUT)
        if audio_frame_count > 0 and not responding:
            log(f"  Silence after {audio_frame_count} frames → processing")
            responding = True
            frames_to_process = audio_frames.copy()
            audio_frames.clear()
            audio_frame_count = 0
            await process_turn(ws, frames_to_process, session_id, device_id, proto_version)
            responding = False

    def reset_silence_timer():
        nonlocal silence_task
        if silence_task and not silence_task.done():
            silence_task.cancel()
        silence_task = asyncio.create_task(on_silence())

    try:
        # Hello handshake
        raw = await asyncio.wait_for(ws.recv(), timeout=15)
        msg = json.loads(raw)
        if msg.get("type") != "hello":
            return

        session_id = msg.get("session_id", "")
        proto_version = msg.get("version", 1)
        device_id = f"device_{addr[0].replace('.', '_')}"
        log(f"  hello v{proto_version} session={session_id} device={device_id}")

        await ws.send(json.dumps({
            "type": "hello",
            "transport": "websocket",
            "session_id": session_id,
            "audio_params": {
                "format": "opus",
                "sample_rate": tts_service.SAMPLE_RATE,
                "channels": tts_service.CHANNELS,
                "frame_duration": tts_service.FRAME_DURATION_MS,
            },
        }))
        log("  >> hello sent")

        # Main loop
        async for message in ws:
            if isinstance(message, bytes):
                if responding:
                    continue
                audio_frames.append(message)
                audio_frame_count += 1

                if audio_frame_count == 1:
                    log(f"  Audio started (mode={listen_mode})")

                if listen_mode == "realtime":
                    if audio_frame_count == MAX_LISTEN_FRAMES:
                        log(f"  Max frames → processing")
                        responding = True
                        frames_to_process = audio_frames.copy()
                        audio_frames.clear()
                        audio_frame_count = 0
                        await process_turn(ws, frames_to_process, session_id, device_id, proto_version)
                        responding = False
                else:
                    reset_silence_timer()

            else:
                try:
                    data = json.loads(message)
                except Exception:
                    continue

                msg_type = data.get("type", "")
                state = data.get("state", "")
                log(f"  << {msg_type} state={state} mode={data.get('mode', '')}")

                if msg_type == "listen":
                    if state == "start":
                        listen_mode = data.get("mode", "auto")
                        audio_frames.clear()
                        audio_frame_count = 0
                    elif state in ("stop", "detect"):
                        if silence_task and not silence_task.done():
                            silence_task.cancel()
                        if not responding and audio_frame_count > 0:
                            responding = True
                            frames_to_process = audio_frames.copy()
                            audio_frames.clear()
                            audio_frame_count = 0
                            await process_turn(ws, frames_to_process, session_id, device_id, proto_version)
                            responding = False

    except websockets.exceptions.ConnectionClosed:
        log(f"[-] Disconnected: {addr}")
    except Exception as e:
        log(f"[!] Error: {type(e).__name__}: {e}")
    finally:
        if silence_task and not silence_task.done():
            silence_task.cancel()


async def main():
    await _init_fallback()
    log(f"WebSocket server listening on {HOST}:{PORT}")
    async with websockets.serve(handler, HOST, PORT):
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
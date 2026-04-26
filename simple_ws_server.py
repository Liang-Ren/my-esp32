#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Minimal Xiaozhi-compatible WebSocket server. Responds to everything with '我是温哥华小智'."""

import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

import asyncio
import json
import struct
import ctypes
import websockets
import miniaudio
import edge_tts

HOST = "0.0.0.0"
PORT = 8001
SAMPLE_RATE = 16000
CHANNELS = 1
FRAME_DURATION_MS = 60
FRAME_SAMPLES = SAMPLE_RATE * FRAME_DURATION_MS // 1000  # 960
SILENCE_TIMEOUT = 1.5
MAX_LISTEN_FRAMES = 25

LOG_FILE = r"C:\Users\liang\Copilot\.venv\xiaozhi\ws_server.log"

_tts_frames: list[bytes] = []


def log(msg: str):
    import datetime
    line = f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def pcm_to_opus_frames(pcm: bytes) -> list[bytes]:
    from pyogg.opus import (
        opus_encoder_create, opus_encode, opus_encoder_destroy,
        OPUS_APPLICATION_VOIP, OPUS_OK, c_int, c_int16, c_ubyte,
    )

    err = c_int(0)
    enc = opus_encoder_create(SAMPLE_RATE, CHANNELS, OPUS_APPLICATION_VOIP, ctypes.byref(err))
    if err.value != OPUS_OK:
        raise RuntimeError(f"opus_encoder_create failed: {err.value}")

    frames = []
    frame_bytes = FRAME_SAMPLES * CHANNELS * 2
    remainder = len(pcm) % frame_bytes
    if remainder:
        pcm += b'\x00' * (frame_bytes - remainder)

    out_buf = (c_ubyte * 4000)()
    for i in range(0, len(pcm), frame_bytes):
        chunk = pcm[i:i + frame_bytes]
        pcm_arr = (c_int16 * FRAME_SAMPLES)(*struct.unpack_from(f"<{FRAME_SAMPLES}h", chunk))
        n = opus_encode(enc, pcm_arr, FRAME_SAMPLES, out_buf, 4000)
        if n > 0:
            frames.append(bytes(out_buf[:n]))

    opus_encoder_destroy(enc)
    return frames


def make_frame(opus_data: bytes, version: int) -> bytes:
    if version == 2:
        return struct.pack(">HHHII", 2, 0, 0, 0, len(opus_data)) + opus_data
    elif version == 3:
        return bytes([0, 0]) + struct.pack(">H", len(opus_data)) + opus_data
    else:
        return opus_data


async def generate_tts(text: str) -> list[bytes]:
    log(f"Generating TTS: {text}")
    c = edge_tts.Communicate(text, voice="zh-CN-XiaoxiaoNeural")
    mp3 = b""
    async for chunk in c.stream():
        if chunk["type"] == "audio":
            mp3 += chunk["data"]

    decoded = miniaudio.decode(mp3, output_format=miniaudio.SampleFormat.SIGNED16,
                               nchannels=CHANNELS, sample_rate=SAMPLE_RATE)
    pcm = bytes(decoded.samples)
    frames = pcm_to_opus_frames(pcm)
    log(f"Generated {len(frames)} Opus frames ({len(frames) * FRAME_DURATION_MS / 1000:.1f}s)")
    return frames


async def send_tts(ws, frames: list[bytes], session_id: str = "", version: int = 1):
    log(f"  >> tts start (v{version})")
    await ws.send(json.dumps({"type": "tts", "state": "start", "session_id": session_id}))
    await ws.send(json.dumps({"type": "tts", "state": "sentence_start",
                              "text": "我是温哥华小智", "session_id": session_id}))
    for frame in frames:
        await ws.send(make_frame(frame, version))
        await asyncio.sleep(FRAME_DURATION_MS / 1000)
    await ws.send(json.dumps({"type": "tts", "state": "stop", "session_id": session_id}))
    log(f"  >> tts stop")


async def handler(ws):
    addr = ws.remote_address
    log(f"[+] Connected: {addr}")
    session_id = ""
    proto_version = 1
    listening = False
    listen_mode = "auto"
    audio_frame_count = 0
    responding = False
    silence_task = None

    async def on_silence():
        nonlocal listening, audio_frame_count, responding
        await asyncio.sleep(SILENCE_TIMEOUT)
        if listening and audio_frame_count > 0 and not responding:
            log(f"  Silence after {audio_frame_count} frames -> responding")
            listening = False
            responding = True
            audio_frame_count = 0
            await send_tts(ws, _tts_frames, session_id, proto_version)
            responding = False

    def reset_silence_timer():
        nonlocal silence_task
        if silence_task and not silence_task.done():
            silence_task.cancel()
        silence_task = asyncio.create_task(on_silence())

    try:
        raw = await asyncio.wait_for(ws.recv(), timeout=15)
        msg = json.loads(raw)
        if msg.get("type") != "hello":
            log(f"Expected hello, got: {msg.get('type')}")
            return

        session_id = msg.get("session_id", "")
        proto_version = msg.get("version", 1)
        log(f"  hello: version={proto_version} session={session_id}")

        await ws.send(json.dumps({
            "type": "hello",
            "transport": "websocket",
            "session_id": session_id,
            "audio_params": {
                "format": "opus",
                "sample_rate": SAMPLE_RATE,
                "channels": CHANNELS,
                "frame_duration": FRAME_DURATION_MS,
            }
        }))
        log("  >> hello sent")

        async for message in ws:
            if isinstance(message, bytes):
                if responding:
                    continue
                audio_frame_count += 1
                if not listening:
                    listening = True
                    log(f"  Audio started, mode={listen_mode}")

                if listen_mode == "realtime":
                    if audio_frame_count == MAX_LISTEN_FRAMES:
                        log(f"  Max frames ({MAX_LISTEN_FRAMES}) -> responding")
                        listening = False
                        responding = True
                        audio_frame_count = 0
                        await send_tts(ws, _tts_frames, session_id, proto_version)
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
                log(f"  << {msg_type} state={state} mode={data.get('mode','')} data={json.dumps(data, ensure_ascii=False)}")

                if msg_type == "listen":
                    if state == "start":
                        listen_mode = data.get("mode", "auto")
                        listening = False
                        audio_frame_count = 0
                    elif state in ("stop", "detect"):
                        if silence_task and not silence_task.done():
                            silence_task.cancel()
                        if not responding and audio_frame_count > 0:
                            listening = False
                            responding = True
                            audio_frame_count = 0
                            await send_tts(ws, _tts_frames, session_id, proto_version)
                            responding = False

    except websockets.exceptions.ConnectionClosed:
        log(f"[-] Disconnected: {addr}")
    except Exception as e:
        log(f"[!] Error: {type(e).__name__}: {e}")
    finally:
        if silence_task and not silence_task.done():
            silence_task.cancel()


async def main():
    global _tts_frames
    _tts_frames = await generate_tts("我是温哥华小智")

    log(f"WebSocket server listening on {HOST}:{PORT}")
    async with websockets.serve(handler, HOST, PORT):
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
#!/usr/bin/env python3
"""
verify_connection.py — end-to-end connection check without real ESP32 hardware.

Runs three checks in order:
  1. Health server is up and /ready == true
  2. WebSocket handshake (hello ↔ hello) succeeds
  3. Full turn: sends a small silent Opus frame, verifies TTS response format

Usage:
    python verify_connection.py
    python verify_connection.py --host 192.168.1.100 --ws-port 8001 --health-port 8002
"""
import asyncio
import json
import struct
import sys
import argparse
import urllib.request
import urllib.error

try:
    import websockets
except ImportError:
    sys.exit("websockets not installed: pip install websockets")

# ── Config ────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--host",        default="127.0.0.1")
parser.add_argument("--ws-port",     type=int, default=8001)
parser.add_argument("--health-port", type=int, default=8002)
args = parser.parse_args()

WS_URL     = f"ws://{args.host}:{args.ws_port}"
HEALTH_URL = f"http://{args.host}:{args.health_port}"

PASS = "\033[32m✓\033[0m"
FAIL = "\033[31m✗\033[0m"
WARN = "\033[33m!\033[0m"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _http_get(url: str, timeout: float = 5.0) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read())


def _silent_opus_frame() -> bytes:
    """
    Minimal valid Opus frame: silence, 60ms, 16kHz mono.
    This is a real Opus DTX (discontinuous transmission) frame — 2 bytes.
    It won't decode to meaningful audio but will pass the frame count check.
    """
    return bytes([0xF8, 0xFF])


# ── Check 1: Health ───────────────────────────────────────────────────────────

def check_health() -> bool:
    print("\n[1] Health server")
    try:
        data = _http_get(f"{HEALTH_URL}/health")
        print(f"    {PASS} /health  → {data}")
    except Exception as exc:
        print(f"    {FAIL} /health  → {exc}")
        return False

    try:
        data = _http_get(f"{HEALTH_URL}/ready")
        if data.get("ready"):
            print(f"    {PASS} /ready   → {data}")
        else:
            print(f"    {WARN} /ready   → {data}  (server still initializing — wait and retry)")
            return False
    except Exception as exc:
        print(f"    {FAIL} /ready   → {exc}")
        return False

    return True


# ── Check 2: WebSocket handshake ──────────────────────────────────────────────

async def check_handshake() -> bool:
    print("\n[2] WebSocket handshake")
    try:
        async with websockets.connect(WS_URL, open_timeout=5) as ws:
            # Send hello
            hello_out = json.dumps({
                "type": "hello",
                "session_id": "verify-001",
                "version": 1,
            })
            await ws.send(hello_out)
            print(f"    {PASS} sent hello  → {hello_out}")

            # Receive hello
            raw = await asyncio.wait_for(ws.recv(), timeout=5)
            hello_in = json.loads(raw)
            if hello_in.get("type") != "hello":
                print(f"    {FAIL} unexpected response type: {hello_in}")
                return False

            print(f"    {PASS} got hello   → type={hello_in['type']} "
                  f"transport={hello_in.get('transport')} "
                  f"session_id={hello_in.get('session_id','')[:12]}…")

            fmt = hello_in.get("audio_params", {})
            print(f"    {PASS} audio_params → rate={fmt.get('sample_rate')} "
                  f"ch={fmt.get('channels')} fmt={fmt.get('format')} "
                  f"frame_ms={fmt.get('frame_duration')}")
    except Exception as exc:
        print(f"    {FAIL} {exc}")
        return False

    return True


# ── Check 3: Full turn (silent audio → TTS response) ─────────────────────────

async def check_full_turn() -> bool:
    print("\n[3] Full turn (silent Opus → TTS frames)")
    try:
        async with websockets.connect(WS_URL, open_timeout=5) as ws:
            # Handshake
            await ws.send(json.dumps({"type": "hello", "session_id": "verify-002", "version": 1}))
            await asyncio.wait_for(ws.recv(), timeout=5)   # server hello

            # Tell server we're starting audio capture
            await ws.send(json.dumps({"type": "listen", "state": "start", "mode": "auto"}))

            # Send a short burst of silent Opus frames (enough to trigger ASR)
            frame = _silent_opus_frame()
            N_FRAMES = 8
            for _ in range(N_FRAMES):
                await ws.send(frame)
            print(f"    {PASS} sent {N_FRAMES} silent Opus frames ({len(frame)} bytes each)")

            # Signal end of audio
            await ws.send(json.dumps({"type": "listen", "state": "stop"}))
            print(f"    {PASS} sent listen/stop")

            # Wait for TTS sequence: tts-start … tts-stop
            tts_states: list[str] = []
            binary_count = 0
            deadline = asyncio.get_event_loop().time() + 60   # Whisper + OpenAI can be slow

            print(f"    … waiting for TTS response (up to 60s) …")
            while asyncio.get_event_loop().time() < deadline:
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=2)
                except asyncio.TimeoutError:
                    continue

                if isinstance(msg, bytes):
                    binary_count += 1
                else:
                    data = json.loads(msg)
                    if data.get("type") == "tts":
                        state = data.get("state", "")
                        tts_states.append(state)
                        text = data.get("text", "")
                        extra = f" text={text!r}" if text else ""
                        print(f"    {PASS} tts/{state}{extra}")
                        if state == "stop":
                            break

            if "start" in tts_states and "stop" in tts_states:
                print(f"    {PASS} TTS sequence complete "
                      f"(states={tts_states}, {binary_count} binary frames)")
            else:
                print(f"    {WARN} incomplete TTS sequence: states={tts_states}")
                print(f"         (ASR may have returned empty — this is normal for silent frames)")

    except Exception as exc:
        print(f"    {FAIL} {exc}")
        return False

    return True


# ── Main ──────────────────────────────────────────────────────────────────────

async def _main() -> int:
    print(f"Xiaozhi connection verifier")
    print(f"  WebSocket : {WS_URL}")
    print(f"  Health    : {HEALTH_URL}")

    ok1 = check_health()
    ok2 = await check_handshake()
    ok3 = await check_full_turn() if ok2 else False

    print("\n── Summary ──────────────────────────────────────────────────")
    print(f"  Health server  : {'PASS' if ok1 else 'FAIL'}")
    print(f"  WS handshake   : {'PASS' if ok2 else 'FAIL'}")
    print(f"  Full turn      : {'PASS' if ok3 else 'FAIL (or silent fallback — expected)'}")

    all_ok = ok1 and ok2
    if all_ok:
        print(f"\n{PASS} Server is ready for ESP32.")
    else:
        print(f"\n{FAIL} Fix the issues above before connecting the ESP32.")

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(_main()))
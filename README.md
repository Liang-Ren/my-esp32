# Xiaozhi ESP32 Backend

WebSocket server for ESP32-S3 AI pod — memory + local Whisper ASR + OpenAI chat.

## Architecture

```
ESP32-S3 (Opus audio)
      │
      │  WebSocket :8001
      ▼
┌─────────────────────────────────────────────────────────────────┐
│  ws_server.py  (entry point)                                    │
│    validate()  ← env check at startup                          │
│    redact()    ← API keys stripped from all log output         │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│  gateway.py  — per-connection handler                           │
│                                                                 │
│  [1] asr.transcribe()          Opus frames → Whisper text      │
│  [2] model_router              detect translation / normal      │
│  [3] memory_service            Mem0 search + user profile      │
│        ├─ Mem0 (cloud/local)   semantic memory                 │
│        └─ SQLite               short-term history + fallback   │
│  [4] prompt_builder            assemble instructions + input   │
│  [5] openai_client             Responses API + thread ID       │
│  [6] response_formatter        strip markdown, cap length      │
│        └─ edge-tts             text → Opus frames              │
│  [7] memory_service.add()      persist turn (non-blocking)     │
│  [8] send TTS frames           JSON + binary → ESP32           │
│                                                                 │
│  Singletons (survive reconnects):                               │
│    _llm        OpenAIClient   previous_response_id threads     │
│    _session_mgr SessionManager                                 │
│    _formatter  ResponseFormatter                               │
└──────────────────────┬──────────────────────────────────────────┘
                       │
         ┌─────────────┴─────────────┐
         │                           │
         ▼                           ▼
  HTTP :8002                  ESP32-S3 (plays audio)
  GET /health                 TTS Opus frames
  GET /ready

OTA server:  HTTP :8000  (ota_server.py — tells ESP32 where to connect)
```

## Files

| File | Purpose |
|------|---------|
| `ws_server.py` | Entry point: env validation, log redaction, start gateway |
| `ota_server.py` | HTTP :8000 — returns WebSocket URL to ESP32 on boot |
| `src/mcp/gateway.py` | WebSocket handler, pipeline, health server |
| `src/mcp/protocol_adapter.py` | ESP32 message parsing + TTS frame building |
| `src/mcp/session_manager.py` | Session lifecycle, device/user ID scoping |
| `src/mcp/response_formatter.py` | Strip markdown, enforce voice length cap |
| `src/ai/openai_client.py` | Responses API wrapper + conversation threading |
| `src/ai/prompt_builder.py` | System prompt + memory context assembly |
| `src/ai/model_router.py` | Detect translation vs normal mode |
| `src/memory/memory_service.py` | Unified Mem0 + SQLite interface |
| `src/memory/mem0_client.py` | Mem0 cloud/self-hosted/mock adapter |
| `src/config/settings.py` | All env vars with validation |
| `src/logging/logger.py` | Structured logs + secret redaction |
| `asr.py` | Opus → faster-whisper (local, free) |
| `tts_service.py` | Text → edge-tts → Opus frames |
| `memory.py` | SQLite short-term + long-term memory |

## Setup

### 1. Install dependencies

```bash
pip install openai faster-whisper edge-tts miniaudio pyogg python-dotenv websockets mem0ai
```

> First run downloads the Whisper `small` model (~500 MB).

### 2. Configure

Copy `.env.example` to `.env` and edit:

```
OPENAI_API_KEY=sk-your-key-here
OPENAI_MODEL=gpt-4o-mini
```

Required fields — the server refuses to start without them:
- `OPENAI_API_KEY`
- `OPENAI_MODEL`

> **Note:** If `OPENAI_API_KEY` is also set as a Windows environment variable,
> `.env` takes precedence (`override=True`).

### 3. Run

OTA server (port 8000) — background:
```bash
pythonw ota_server.py
```

WebSocket server (port 8001) + health server (port 8002):
```bash
python ws_server.py
```

Auto-start shortcuts are in `C:\Users\liang\AppData\Local\XiaozhiOTA\`.

### 4. Verify startup

```
GET http://localhost:8002/health  → {"status":"ok","uptime_s":N}
GET http://localhost:8002/ready   → {"ready":true}
```

`/ready` returns `503` until the Whisper model is loaded and fallback TTS is generated.

### 5. Test without hardware

```bash
python simulate_flow.py
```

Type messages, see AI responses. Memory saved to `memory.db`.

---

## Deployment

### Local (LAN demo)

The OTA server at `:8000` tells the ESP32 where to connect. Edit `ota_server.py`:
```python
WS_URL = "ws://YOUR_LOCAL_IP:8001"
```

Run both servers. The ESP32 fetches the URL on boot and connects automatically.

### Remote access without port-forwarding

**Tailscale** (recommended for persistent demos):
1. Install Tailscale on the server machine and the demo laptop.
2. Set `WS_URL = "ws://100.x.x.x:8001"` (Tailscale IP).
3. No firewall rules needed.

**ngrok** (quick throwaway tunnel):
```bash
ngrok tcp 8001
```
Use the `tcp://X.tcp.ngrok.io:PORT` URL in `ota_server.py` as `ws://X.tcp.ngrok.io:PORT`.

### Cloud VPS (Ubuntu / Debian)

1. Clone repo, create `.env`, install deps.
2. Create a systemd service:

```ini
# /etc/systemd/system/xiaozhi.service
[Unit]
Description=Xiaozhi WebSocket Server
After=network.target

[Service]
WorkingDirectory=/opt/xiaozhi
ExecStart=/usr/bin/python3 ws_server.py
Restart=on-failure
RestartSec=5
StandardOutput=journal
StandardError=journal
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable --now xiaozhi
sudo journalctl -u xiaozhi -f   # live logs
```

3. Allow ports 8001 (WebSocket) and 8002 (health) in your firewall:
```bash
sudo ufw allow 8001/tcp
sudo ufw allow 8002/tcp
```

4. Use a process supervisor or `@reboot` cron entry to start the OTA server
   (`pythonw ota_server.py` or `python3 ota_server.py &`).

### Environment variables reference

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | — | **Required.** OpenAI API key. |
| `OPENAI_MODEL` | `gpt-4o-mini` | Model name passed to Responses API. |
| `LLM_TIMEOUT` | `30.0` | Seconds before the OpenAI call is abandoned. |
| `MEM0_API_KEY` | — | Mem0 cloud key (or leave blank for self-hosted). |
| `MEM0_SERVER_URL` | — | Self-hosted Mem0 URL (or leave blank for mock). |
| `MEMORY_TIMEOUT` | `5.0` | Seconds before memory lookup is skipped. |
| `WS_PORT` | `8001` | WebSocket listen port. |
| `WS_PING_INTERVAL` | `20` | Seconds between server pings to ESP32. |
| `WS_PING_TIMEOUT` | `20` | Seconds to wait for pong before disconnect. |
| `HEALTH_PORT` | `8002` | HTTP health server port. |
| `SILENCE_TIMEOUT` | `1.5` | Seconds of silence before processing audio. |
| `MAX_LISTEN_FRAMES` | `25` | Opus frames before forced transcription. |
| `MAX_VOICE_REPLY_CHARS` | `300` | Character cap on voice replies. |

---

## Testing

No real API keys or hardware required. All external services are mocked.

### Install test dependencies

```bash
pip install pytest pytest-asyncio
```

### Run all tests

```bash
python -m pytest
```

### Run a specific file

```bash
python -m pytest tests/test_protocol.py    # ESP32 message parsing + response format
python -m pytest tests/test_memory.py     # Mem0 fallback + multi-device isolation
python -m pytest tests/test_pipeline.py   # full request-flow integration
python -m pytest tests/test_threading.py  # previous_response_id conversation threading
python -m pytest tests/test_hardening.py  # env validation, redaction, health, timeouts
```

### Test coverage summary

| File | Tests | What's covered |
|------|-------|----------------|
| `test_protocol.py`  | 17 | hello/listen parsing (v1–v3), TTS JSON schema, Opus headers |
| `test_memory.py`    | 11 | Mem0 down → SQLite fallback, per-device isolation |
| `test_pipeline.py`  | 18 | memory-before-LLM ordering, prompt injection, fallback TTS |
| `test_threading.py` | 15 | `previous_response_id`, stale-ID retry, `clear_thread` |
| `test_hardening.py` | 25 | env validation, log redaction, health endpoint, memory timeout |

---

## Model routing

| Trigger | Behaviour |
|---------|-----------|
| "翻译" / "translate" | Short translation-only prompt, no memory |
| Normal speech | Full prompt with memory context |
| Empty ASR / failed | Fallback: "我没听清楚，请再说一遍。" |

## Logs

- `server.log` — structured JSON per request + step-level timing
- `ws_stderr.txt` / `ws_stdout.txt` — live output (when started via redirect)
- `memory.db` — SQLite database (short-term history + long-term profile)

API keys are **never** written to any log file — they are redacted to `[REDACTED]`
at the logging layer before any handler receives the record.

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `[STARTUP ERROR] OPENAI_API_KEY is required` | Missing env var | Add key to `.env` |
| ESP32 "无法连接服务" | OTA returns wrong IP | Edit `WS_URL` in `ota_server.py` |
| 反复"我没听清楚" | ASR error | Check `ws_stdout.txt` for traceback |
| Chat 429 insufficient_quota | Wrong key | Verify `.env` has correct key |
| `/ready` stays `false` | Whisper model loading | Wait 30–60s on first run |
| ws_server exits immediately | Port 8001 in use | Kill old process: `netstat -ano \| findstr :8001` |

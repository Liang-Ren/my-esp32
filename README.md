# Xiaozhi ESP32 Backend

WebSocket server for the Xiaozhi ESP32-S3 AI voice pod.

```
ESP32-S3  ──Opus audio──►  ws_server.py :8001  ──►  OpenAI (LLM + Whisper)
                                │                         │
                           health :8002            OpenMemory :8765
                                                   (long-term memory)
OTA server :8000  ◄──  ESP32 boot  (tells device where to connect)
```

---

## Quick start

### 1. Install Python dependencies

```bash
pip install openai faster-whisper edge-tts miniaudio pyogg python-dotenv websockets mem0ai httpx
```

### 2. Configure `.env`

Copy the example and fill in your values:

```bash
cp .env.example .env
```

Minimum required:

```env
OPENAI_API_KEY=sk-...        # Required — LLM + Whisper API
OPENAI_MODEL=gpt-4o-mini     # Or gpt-4o, etc.
```

Memory backend — pick one (or leave both blank to use the in-process mock):

```env
# Option A: self-hosted OpenMemory (recommended, see step 3)
MEM0_SERVER_URL=http://localhost:8765

# Option B: Mem0 cloud
MEM0_API_KEY=m0-...
```

Full reference at the bottom of this file.

### 3. Start OpenMemory (self-hosted memory, optional)

Requires Docker. Run once from the `openmemory/` directory:

```bash
cd C:\path\to\mem0-src\openmemory
docker compose up -d
```

This starts two containers:
- `qdrant` on port 6333 — vector store
- `openmemory-mcp` on port 8765 — memory API

Verify it's up:

```
GET http://localhost:8765/openapi.json  →  200
```

### 4. Configure the OTA server

Edit `ota_server.py` and set your machine's LAN IP:

```python
WS_URL = "ws://YOUR_LAN_IP:8001"   # e.g. ws://192.168.1.10:8001
```

The ESP32 fetches this URL on boot to know where to connect.

### 5. Start all servers

Open two terminals in the project directory.

**Terminal 1 — OTA server** (stays in background, restart rarely needed):

```bash
pythonw ota_server.py         # Windows — runs without a console window
# or
python ota_server.py          # any platform
```

**Terminal 2 — WebSocket + health server**:

```bash
python ws_server.py
```

Expected output:

```
[HH:MM:SS] LLM: gpt-4o-mini (timeout=30.0s)
[HH:MM:SS] Fallback TTS ready (48 frames)
[HH:MM:SS] WebSocket on 0.0.0.0:8001 (ping every 20s)
server listening on 0.0.0.0:8001
[HH:MM:SS] Health on 0.0.0.0:8002
```

### 6. Verify before connecting the ESP32

```bash
python verify_connection.py
```

This checks health endpoint → WebSocket handshake → TTS round-trip without hardware.

### 7. Connect the ESP32

Reboot the ESP32. It fetches the OTA URL, connects to `:8001`, and is ready to talk.

---

## Architecture

```
ESP32-S3 (Opus audio, protocol v1)
      │  WebSocket :8001
      ▼
┌─────────────────────────────────────────────────────────┐
│  ws_server.py  — startup validation, log redaction      │
└──────────────────────┬──────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────┐
│  src/mcp/gateway.py  — per-connection pipeline          │
│                                                         │
│  [1] asr.py              Opus → OpenAI Whisper → text  │
│  [2] model_router        detect translation / normal    │
│  [3] memory_service      OpenMemory search + profile    │
│  [4] prompt_builder      assemble system prompt         │
│  [5] openai_client       Responses API, threaded        │
│  [6] response_formatter  strip markdown, cap length     │
│        └─ tts_service    text → edge-tts → Opus        │
│  [7] memory_service      persist turn (non-blocking)   │
│  [8] send TTS frames     JSON + binary → ESP32          │
└──────────────────────┬──────────────────────────────────┘
         ┌─────────────┴──────────────┐
         ▼                            ▼
  HTTP :8002                   OpenMemory :8765
  /health  /ready              (Docker — Qdrant + API)
```

---

## File reference

| File | Purpose |
|------|---------|
| `ws_server.py` | Entry point: validation, redaction, start gateway |
| `ota_server.py` | HTTP :8000 — returns WebSocket URL to ESP32 on boot |
| `src/mcp/gateway.py` | WebSocket handler, full pipeline, health server |
| `src/mcp/protocol_adapter.py` | ESP32 message parsing, TTS frame building |
| `src/mcp/session_manager.py` | Session lifecycle, device/user ID scoping |
| `src/mcp/response_formatter.py` | Strip markdown, enforce voice length cap |
| `src/ai/openai_client.py` | Responses API wrapper + conversation threading |
| `src/ai/prompt_builder.py` | System prompt + memory context assembly |
| `src/ai/model_router.py` | Detect translation vs normal mode |
| `src/memory/memory_service.py` | Unified OpenMemory + SQLite interface |
| `src/memory/mem0_client.py` | OpenMemory HTTP adapter / Mem0 cloud / mock |
| `src/config/settings.py` | All env vars with validation |
| `asr.py` | Opus decode → OpenAI Whisper API (local faster-whisper fallback) |
| `tts_service.py` | Text → edge-tts → Opus frames (zh-CN-XiaoxiaoNeural) |
| `memory.py` | SQLite short-term history + long-term facts |
| `verify_connection.py` | End-to-end connection test without hardware |

---

## Environment variables

### Required

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | OpenAI API key — used for both LLM (Responses API) and ASR (Whisper) |
| `OPENAI_MODEL` | Model name, e.g. `gpt-4o-mini` or `gpt-4o` |

### Memory backend (pick one or neither)

| Variable | Description |
|----------|-------------|
| `MEM0_SERVER_URL` | Self-hosted OpenMemory URL, e.g. `http://localhost:8765` |
| `MEM0_API_KEY` | Mem0 cloud API key (`m0-...`) — takes priority over `MEM0_SERVER_URL` |

If neither is set, an in-process mock is used (memory lost on restart).

### Optional tuning

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_TIMEOUT` | `30.0` | Seconds before the OpenAI call is abandoned |
| `MEMORY_TIMEOUT` | `5.0` | Seconds before memory lookup is skipped |
| `WS_PORT` | `8001` | WebSocket listen port |
| `WS_PING_INTERVAL` | `20` | Seconds between WebSocket pings to ESP32 |
| `WS_PING_TIMEOUT` | `60` | Seconds to wait for pong before disconnect |
| `HEALTH_PORT` | `8002` | HTTP health server port |
| `MAX_LISTEN_FRAMES` | `50` | Max Opus frames collected before forced transcription (50 × 60ms = 3s) |
| `SILENCE_TIMEOUT` | `1.5` | Seconds of silence before processing (auto/manual modes) |
| `MAX_VOICE_REPLY_CHARS` | `300` | Character cap on voice replies |

---

## Testing

No API keys or hardware required — all external services are mocked.

```bash
pip install pytest pytest-asyncio
python -m pytest                          # all tests
python -m pytest tests/test_protocol.py  # ESP32 message parsing
python -m pytest tests/test_memory.py    # memory fallback + isolation
python -m pytest tests/test_pipeline.py  # full request flow
python -m pytest tests/test_threading.py # conversation threading
python -m pytest tests/test_hardening.py # validation, redaction, timeouts
```

---

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `[STARTUP ERROR] OPENAI_API_KEY is required` | Missing `.env` | Add key to `.env` |
| ESP32 "无法连接服务" | OTA returns wrong IP | Edit `WS_URL` in `ota_server.py` |
| 反复"我没听清楚" | ASR returning empty | Check `ws_stderr.txt` for errors |
| `/ready` stays `false` | Whisper local model loading | Wait 30–60s on first run (only if no `OPENAI_API_KEY`) |
| Port 8001 in use | Old process still running | `netstat -ano \| findstr :8001` then kill the PID |
| Frequent disconnects during TTS | Ping timeout too short | Set `WS_PING_TIMEOUT=60` in `.env` |
| Memory not persisting across restarts | No memory backend configured | Add `MEM0_SERVER_URL` or `MEM0_API_KEY` to `.env` |

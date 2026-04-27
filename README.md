# Xiaozhi ESP32 Backend

WebSocket server for ESP32-S3 AI pod — memory + local Whisper ASR + OpenAI chat.

## Files

| File | Purpose |
|------|---------|
| `ws_server.py` | Main WebSocket server (ESP32 connects here) |
| `ota_server.py` | HTTP server telling ESP32 where to connect |
| `asr.py` | Opus audio → faster-whisper (local) → text |
| `llm.py` | OpenAI ChatGPT client + routing |
| `prompt_builder.py` | System prompt + memory + history builder |
| `memory.py` | SQLite short-term + long-term memory |
| `tts_service.py` | Text → edge-tts → Opus frames |
| `logger.py` | Structured logging (console + server.log) |
| `simulate.py` | Local text-only test (no audio needed) |

## Setup

### 1. Install dependencies

```bash
pip install openai faster-whisper edge-tts miniaudio pyogg python-dotenv websockets
```

> First run will download the Whisper `small` model (~500 MB).

### 2. Configure secrets

Edit `.env`:
```
OPENAI_API_KEY=sk-your-key-here
OPENAI_MODEL=gpt-4o-mini
```

> **Note:** If `OPENAI_API_KEY` is also set as a Windows environment variable, the `.env` file takes precedence (scripts use `override=True`).

### 3. Run servers

OTA server (port 8000) — background, no console needed:
```bash
pythonw ota_server.py
```

WebSocket server (port 8001) — must use `python`, not `pythonw`:
```bash
python ws_server.py
```

Auto-start shortcuts are in `C:\Users\liang\AppData\Local\XiaozhiOTA\`.

### 4. Test without hardware

```bash
python simulate.py
```

Type messages and see AI responses. Memory is saved to `memory.db`.

### 5. Test API key

```bash
python test_openai.py
```

## Architecture

```
ESP32 (Opus audio)
  → ws_server.py       hello handshake + audio collection
  → asr.py             Opus → PCM → WAV → faster-whisper (local, free)
  → prompt_builder.py  system + long-term memory + history + input
  → llm.py             OpenAI gpt-4o-mini → response text
  → memory.py          save to SQLite (short + long term)
  → tts_service.py     text → edge-tts → Opus frames
  → ESP32 (plays audio)
```

## Logs

- `server.log` — structured JSON per request (request_id, device_id, latency, tokens, errors)
- `ws_stderr.txt` / `ws_stdout.txt` — live server output (when started via python.exe redirect)
- `memory.db` — SQLite database

## Model routing

| Trigger | Behaviour |
|---------|-----------|
| "翻译" / "translate" | Short translation-only prompt |
| Normal speech | Full prompt with memory |
| ASR empty / failed | Safe fallback: "我没听清楚，请再说一遍。" |

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| ESP32 "无法连接服务" | OTA returns wrong IP | Check `WS_URL` in `ota_server.py` |
| 反复"我没听清楚" | ASR error | Check `ws_stdout.txt` for traceback |
| Chat 429 insufficient_quota | Wrong API key loaded | Ensure `.env` has correct key; Windows env var may override |
| ws_server exits immediately | Port 8001 in use | Kill old process first |
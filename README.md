# Xiaozhi ESP32 Backend

WebSocket server for ESP32-S3 AI pod — memory + OpenAI Q&A.

## Files

| File | Purpose |
|------|---------|
| `ws_server.py` | Main WebSocket server (ESP32 connects here) |
| `ota_server.py` | HTTP server telling ESP32 where to connect |
| `asr.py` | Opus audio → OpenAI Whisper → text |
| `llm.py` | OpenAI ChatGPT client + routing |
| `prompt_builder.py` | System prompt + memory + history builder |
| `memory.py` | SQLite short-term + long-term memory |
| `tts_service.py` | Text → edge-tts → Opus frames |
| `logger.py` | Structured logging (console + server.log) |
| `simulate.py` | Local text-only test (no audio needed) |

## Setup

### 1. Install dependencies

```bash
pip install openai edge-tts miniaudio pyogg python-dotenv websockets
```

### 2. Configure secrets

Edit `.env`:
```
OPENAI_API_KEY=sk-your-key-here
OPENAI_MODEL=gpt-4o-mini
```

### 3. Run servers

OTA server (port 8000):
```bash
pythonw ota_server.py
```

WebSocket server (port 8001):
```bash
python ws_server.py
```

### 4. Test without hardware

```bash
python simulate.py
```

Type messages and see AI responses. Memory is saved to `memory.db`.

## Architecture

```
ESP32 (Opus audio)
  → ws_server.py       hello handshake + audio collection
  → asr.py             Opus → PCM → WAV → Whisper → text
  → prompt_builder.py  system + long-term memory + history + input
  → llm.py             OpenAI gpt-4o-mini → response text
  → memory.py          save to SQLite (short + long term)
  → tts_service.py     text → edge-tts → Opus frames
  → ESP32 (plays audio)
```

## Logs

- `server.log` — structured JSON per request (request_id, device_id, latency, tokens, errors)
- `ws_server.log` — connection events (legacy, still written by logger)
- `memory.db` — SQLite database

## Model routing

| Trigger | Behaviour |
|---------|-----------|
| "翻译" / "translate" | Short translation-only prompt |
| Normal speech | Full prompt with memory |
| ASR empty / failed | Safe fallback: "我没听清楚，请再说一遍。" |
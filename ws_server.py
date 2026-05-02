#!/usr/bin/env python3
"""
ws_server.py — entry point (kept for autostart compatibility).
All logic lives in src/mcp/gateway.py.
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

import asyncio
from src.config.settings import settings
from src.logging.logger import redact
from src.mcp.gateway import run

if __name__ == "__main__":
    # Register API keys for log redaction before anything else runs.
    redact(settings.OPENAI_API_KEY)
    redact(settings.MEM0_API_KEY)

    try:
        settings.validate()
    except ValueError as exc:
        print(f"[STARTUP ERROR]\n{exc}", file=sys.stderr)
        sys.exit(1)

    asyncio.run(run())

#!/usr/bin/env python3
"""
ws_server.py — entry point (kept for autostart compatibility).
All logic lives in src/mcp/gateway.py.
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

import asyncio
from src.mcp.gateway import run

if __name__ == "__main__":
    print(f"server listening on 0.0.0.0:8001")
    asyncio.run(run())
#!/usr/bin/env python3
import sys, asyncio
sys.stdout.reconfigure(encoding="utf-8")

from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env", override=True)

import os
from openai import AsyncOpenAI

API_KEY = os.getenv("OPENAI_API_KEY", "")
MODEL   = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

print(f"Key : {API_KEY[:8]}...{API_KEY[-4:]}")
print(f"Model: {MODEL}")
print()

async def test():
    client = AsyncOpenAI(api_key=API_KEY)
    try:
        resp = await client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": "回复'OK'两个字"}],
            max_tokens=10,
        )
        print(f"Chat OK: {resp.choices[0].message.content}")
        print(f"Tokens used: {resp.usage.total_tokens}")
    except Exception as e:
        print(f"Chat FAIL: {e}")

asyncio.run(test())
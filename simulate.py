#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Local simulation: type messages, get AI responses. No audio required."""

import sys
sys.stdout.reconfigure(encoding="utf-8")

import asyncio
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

from memory import Memory
from prompt_builder import build_messages, detect_mode
from llm import LLMClient

DEVICE_ID = "simulate_001"


async def run():
    memory = Memory()
    llm = LLMClient()

    print("=== Xiaozhi Simulator ===")
    print(f"Device: {DEVICE_ID}  |  type 'quit' to exit\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if user_input.lower() in ("quit", "exit", "q"):
            break
        if not user_input:
            continue

        history = memory.get_recent(DEVICE_ID)
        long_term = memory.get_long_term(DEVICE_ID)
        mode = detect_mode(user_input)
        messages = build_messages(user_input, history, long_term, mode)

        print(f"[mode={mode}, history={len(history)} msgs]")
        response, usage = await llm.complete(messages)

        memory.add_message(DEVICE_ID, "user", user_input)
        memory.add_message(DEVICE_ID, "assistant", response)

        print(f"AI : {response}")
        tokens = usage.get("total_tokens", "?")
        model = usage.get("model", "?")
        error = usage.get("error")
        if error:
            print(f"    [ERROR: {error}]")
        else:
            print(f"    [tokens={tokens}, model={model}]")
        print()


if __name__ == "__main__":
    asyncio.run(run())
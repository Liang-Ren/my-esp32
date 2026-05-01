#!/usr/bin/env python3
"""
simulate_v2.py — text-only simulation of the refactored pipeline.
No ESP32 or audio hardware required.

Usage:
    python simulate_v2.py
"""
import sys, asyncio
sys.stdout.reconfigure(encoding="utf-8")

from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env", override=True)

from src.config.settings import settings
from src.memory.mem0_client import Mem0Client, MockMem0Client
from src.memory.memory_service import MemoryService
from src.ai.openai_client import OpenAIClient
from src.ai.model_router import ModelRouter
from src.ai.prompt_builder import build_input

from memory import Memory as SqliteMemory

DEVICE_ID = "simulate_v2"
USER_ID = f"{settings.MEM0_USER_ID_PREFIX}{DEVICE_ID}"


def _build_mem0():
    if settings.MEM0_API_KEY or settings.MEM0_SERVER_URL:
        print(f"Mem0: {'cloud' if settings.MEM0_API_KEY else 'self-hosted'}")
        return Mem0Client(settings.MEM0_API_KEY, settings.MEM0_SERVER_URL)
    print("Mem0: not configured — using in-process mock")
    return MockMem0Client()


async def run() -> None:
    mem0 = _build_mem0()
    sqlite = SqliteMemory()
    memory_svc = MemoryService(mem0_client=mem0, sqlite_memory=sqlite)
    llm = OpenAIClient(settings.OPENAI_API_KEY, settings.OPENAI_MODEL)
    router = ModelRouter()

    print("=== Xiaozhi Simulator v2 ===")
    print(f"Device: {DEVICE_ID}  |  User: {USER_ID}")
    print(f"Model:  {settings.OPENAI_MODEL}")
    print("Type 'quit' to exit\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if user_input.lower() in ("quit", "exit", "q"):
            break
        if not user_input:
            continue

        memories, profile = await asyncio.gather(
            memory_svc.getRelevantMemories(USER_ID, DEVICE_ID, user_input),
            memory_svc.getUserProfile(USER_ID, DEVICE_ID),
        )
        history = memory_svc.get_history(DEVICE_ID)
        mode = router.detect_mode(user_input)

        instructions, input_msgs = build_input(
            user_input, history,
            memory_summary=profile["summary"],
            recent_memory=memories,
            user_preferences=profile["preferences_str"],
            mode=mode,
        )

        print(f"  [mode={mode}, history={len(history)}, "
              f"memories={len(memories)}, summary={bool(profile['summary'])}]")

        response, usage = await llm.generateResponse(input_msgs, context=instructions)
        await memory_svc.addInteraction(USER_ID, DEVICE_ID, user_input, response)

        print(f"AI : {response}")
        if usage.get("error"):
            print(f"  [ERROR: {usage['error']}] ({usage.get('latency_ms')}ms)")
        else:
            print(f"  [tokens={usage.get('total_tokens','?')}, "
                  f"latency={usage.get('latency_ms')}ms, "
                  f"id={usage.get('response_id','?')[:16]}...]")
        print()


if __name__ == "__main__":
    asyncio.run(run())

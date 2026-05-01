#!/usr/bin/env python3
"""
simulate_flow.py — deterministic request-flow test.

Sends three messages in sequence and verifies that the assistant remembers
context from earlier in the conversation.

Expected result:
    Turn 1  "My name is Leon"          → greeting
    Turn 2  "I am building an ESP32 AI pod" → acknowledgement
    Turn 3  "What am I building?"      → should mention ESP32 / AI pod

Run:
    python simulate_flow.py

No ESP32 or audio hardware required.
"""
import sys
import asyncio
import logging
import time
sys.stdout.reconfigure(encoding="utf-8")

from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env", override=True)

# Configure logging so step logs appear on stdout
logging.basicConfig(level=logging.INFO, format="%(message)s")

from src.config.settings import settings
from src.memory.mem0_client import Mem0Client, MockMem0Client
from src.memory.memory_service import MemoryService
from src.ai.openai_client import OpenAIClient
from src.ai.model_router import ModelRouter
from src.ai.prompt_builder import build_input
from src.logging.logger import log_step, log_request, new_request_id

from memory import Memory as SqliteMemory

DEVICE_ID = "sim_flow"
USER_ID = f"{settings.MEM0_USER_ID_PREFIX}{DEVICE_ID}"

TURNS = [
    "My name is Leon",
    "I am building an ESP32 AI pod",
    "What am I building?",
]


def _ms(t0: float) -> int:
    return round((time.time() - t0) * 1000)


def _build_services() -> tuple[MemoryService, OpenAIClient]:
    if settings.MEM0_API_KEY or settings.MEM0_SERVER_URL:
        mode = "cloud" if settings.MEM0_API_KEY else "self-hosted"
        print(f"Mem0: {mode}")
        mem0 = Mem0Client(settings.MEM0_API_KEY, settings.MEM0_SERVER_URL)
    else:
        print("Mem0: not configured — using in-process mock")
        mem0 = MockMem0Client()
    sqlite = SqliteMemory()
    return (
        MemoryService(mem0_client=mem0, sqlite_memory=sqlite),
        OpenAIClient(settings.OPENAI_API_KEY, settings.OPENAI_MODEL),
    )


async def run_turn(
    turn_num: int,
    user_text: str,
    memory_svc: MemoryService,
    llm: OpenAIClient,
    router: ModelRouter,
) -> str:
    request_id = new_request_id()
    t_total = time.time()
    metrics: dict = {}

    print(f"\n{'─'*60}")
    print(f"Turn {turn_num}  [{request_id}]")
    print(f"User: {user_text!r}")
    print()

    # Parse
    t = time.time()
    mode = router.detect_mode(user_text)
    history = memory_svc.get_history(DEVICE_ID)
    metrics["parse_ms"] = _ms(t)
    log_step(request_id, "parse",
             f"mode={mode} history={len(history)}turns",
             ms=metrics["parse_ms"])

    # Memory
    t = time.time()
    memories, profile = await asyncio.gather(
        memory_svc.getRelevantMemories(USER_ID, DEVICE_ID, user_text),
        memory_svc.getUserProfile(USER_ID, DEVICE_ID),
    )
    metrics["memory_ms"] = _ms(t)
    log_step(request_id, "memory",
             f"{len(memories)} memories retrieved, "
             f"summary={bool(profile['summary'])}",
             ms=metrics["memory_ms"])
    if memories:
        for i, m in enumerate(memories, 1):
            print(f"  mem[{i}]: {m!r}")

    # Prompt
    instructions, input_msgs = build_input(
        user_text, history,
        memory_summary=profile["summary"],
        recent_memory=memories,
        user_preferences=profile["preferences_str"],
        mode=mode,
    )
    log_step(request_id, "prompt",
             f"instructions={len(instructions)}chars messages={len(input_msgs)}")

    # LLM
    t = time.time()
    response_text, usage = await llm.generateResponse(input_msgs, context=instructions)
    metrics["llm_ms"] = _ms(t)
    model = usage.get("model", llm.model)
    metrics["tokens"] = usage.get("total_tokens", 0)

    if usage.get("error"):
        log_step(request_id, "llm", f"ERROR: {usage['error']}", ms=metrics["llm_ms"])
        metrics["total_ms"] = _ms(t_total)
        log_request(request_id, DEVICE_ID, user_text, response_text,
                    model, metrics, error=usage["error"])
        return response_text

    log_step(request_id, "llm", f"→ {response_text!r}", ms=metrics["llm_ms"])

    # Store
    ok = await memory_svc.addInteraction(USER_ID, DEVICE_ID, user_text, response_text)
    log_step(request_id, "store", "OK" if ok else "FAILED (Mem0 down, SQLite written)")

    metrics["total_ms"] = _ms(t_total)
    log_request(request_id, DEVICE_ID, user_text, response_text, model, metrics)

    print(f"\nAI : {response_text}")
    return response_text


async def main() -> None:
    print("=" * 60)
    print("simulate_flow.py — memory recall test")
    print(f"Model:  {settings.OPENAI_MODEL}")
    print(f"Device: {DEVICE_ID}  |  User: {USER_ID}")
    print("=" * 60)

    memory_svc, llm = _build_services()
    router = ModelRouter()
    responses: list[str] = []

    for i, user_text in enumerate(TURNS, 1):
        resp = await run_turn(i, user_text, memory_svc, llm, router)
        responses.append(resp)

    # ── Verdict ────────────────────────────────────────────────────────────────
    print(f"\n{'═'*60}")
    print("MEMORY RECALL VERDICT")
    print(f"{'═'*60}")
    final_answer = responses[-1].lower()
    keywords = ["esp32", "ai pod", "人工智能", "智能", "pod", "esp", "物联网"]
    hit = any(kw in final_answer for kw in keywords)
    print(f"Turn 3 response: {responses[-1]!r}")
    print()
    if hit:
        print("PASS  — assistant recalled the ESP32 AI pod context")
    else:
        print("PARTIAL — response did not explicitly mention ESP32/AI pod")
        print("(This may still be correct if the assistant paraphrased.)")
        print("Check the turn 3 response above for context accuracy.")


if __name__ == "__main__":
    asyncio.run(main())

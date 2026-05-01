"""
MemoryService — unified interface over Mem0 and SQLite.

Mem0 is the primary store for semantic memory. SQLite is always written
and serves as the local fallback when Mem0 is unreachable.

Gateway calls:
    getRelevantMemories(user_id, device_id, user_text) -> list[str]
    addInteraction(user_id, device_id, user_text, assistant_reply) -> bool
    getUserProfile(user_id, device_id) -> dict
    updateImportantFacts(user_id, device_id, facts) -> None
    get_history(device_id) -> list[dict]   # short-term turns for LLM context
"""
import time
import logging

log = logging.getLogger("xiaozhi.src")

_FALLBACK_TOP_K = 5


class MemoryService:
    def __init__(self, mem0_client=None, sqlite_memory=None):
        self._mem0 = mem0_client     # Mem0Client | MockMem0Client | None
        self._sqlite = sqlite_memory  # legacy memory.Memory | None

    # ── Primary interface ──────────────────────────────────────────────────────

    async def getRelevantMemories(
        self, user_id: str, device_id: str, user_text: str
    ) -> list[str]:
        """
        Semantic search against Mem0 for memories relevant to user_text.

        Falls back to SQLite facts if Mem0 is unavailable.
        Logs lookup latency and result count.

        Returns a list of plain-text memory strings.
        """
        t0 = time.time()
        memories: list[str] = []
        source = "none"

        if self._mem0:
            try:
                hits = await self._mem0.search(
                    user_text, user_id, top_k=_FALLBACK_TOP_K
                )
                memories = [
                    h.get("memory") or h.get("text") or ""
                    for h in hits
                    if h.get("memory") or h.get("text")
                ]
                source = "mem0"
            except Exception as exc:
                log.warning("Mem0 search failed (%s) — falling back to SQLite", exc)
                memories = self._sqlite_facts(device_id)
                source = "sqlite-fallback"
        elif self._sqlite:
            memories = self._sqlite_facts(device_id)
            source = "sqlite"

        latency_ms = round((time.time() - t0) * 1000)
        log.info(
            "  Memory lookup: %d result(s) in %dms [%s]",
            len(memories), latency_ms, source,
        )
        return memories

    async def addInteraction(
        self,
        user_id: str,
        device_id: str,
        user_text: str,
        assistant_reply: str,
    ) -> bool:
        """
        Persist one conversation turn.

        SQLite is always written (synchronous, reliable short-term history).
        Mem0 write is attempted after; failure is logged but not re-raised.

        Returns True if Mem0 write succeeded (or Mem0 not configured).
        """
        messages = [
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": assistant_reply},
        ]

        # SQLite — always
        if self._sqlite:
            self._sqlite.add_message(device_id, "user", user_text)
            self._sqlite.add_message(device_id, "assistant", assistant_reply)

        # Mem0 — best-effort
        if self._mem0:
            try:
                await self._mem0.add(messages, user_id)
                log.info("  Memory write: OK [mem0 user=%s]", user_id)
                return True
            except Exception as exc:
                log.warning("  Memory write: FAILED (%s)", exc)
                return False

        return True

    async def getUserProfile(
        self, user_id: str, device_id: str
    ) -> dict:
        """
        Return user profile from SQLite long-term store.

        Keys:
            summary         str  — freeform user background text
            facts           list[str] — remembered facts
            preferences     dict — key/value preference pairs
            preferences_str str  — comma-joined "k:v" string for prompt injection
        """
        profile = {
            "summary": "",
            "facts": [],
            "preferences": {},
            "preferences_str": "",
        }
        if self._sqlite:
            lt = self._sqlite.get_long_term(device_id)
            profile["summary"] = lt.get("summary", "")
            profile["facts"] = lt.get("facts", [])
            prefs = lt.get("preferences", {})
            if isinstance(prefs, dict):
                profile["preferences"] = prefs
                if prefs:
                    profile["preferences_str"] = "、".join(
                        f"{k}:{v}" for k, v in prefs.items()
                    )
        return profile

    async def updateImportantFacts(
        self, user_id: str, device_id: str, facts: list[str]
    ) -> None:
        """
        Overwrite the long-term facts list.

        Writes to SQLite immediately and queues a Mem0 add if configured.
        Call this from application logic whenever new facts are extracted
        (e.g., after every N interactions or via an explicit extraction step).
        """
        if self._sqlite:
            self._sqlite.update_long_term(device_id, facts=facts)

        if self._mem0 and facts:
            try:
                facts_msg = [
                    {
                        "role": "system",
                        "content": "Important facts about this user: "
                        + "; ".join(facts),
                    }
                ]
                await self._mem0.add(facts_msg, user_id)
                log.info("  updateImportantFacts: synced %d fact(s) to Mem0", len(facts))
            except Exception as exc:
                log.warning("  updateImportantFacts: Mem0 write failed (%s)", exc)

    # ── Short-term history (for LLM input) ────────────────────────────────────

    def get_history(self, device_id: str) -> list[dict]:
        """Last N conversation turns from SQLite (used as LLM input context)."""
        if self._sqlite:
            return self._sqlite.get_recent(device_id)
        return []

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _sqlite_facts(self, device_id: str) -> list[str]:
        """Pull stored facts from SQLite long-term memory."""
        if not self._sqlite:
            return []
        lt = self._sqlite.get_long_term(device_id)
        return lt.get("facts", [])[:_FALLBACK_TOP_K]

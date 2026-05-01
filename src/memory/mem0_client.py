"""
Mem0 memory backend.

Two production modes (set one in .env):
  MEM0_API_KEY     — Mem0 cloud (https://mem0.ai)
  MEM0_SERVER_URL  — Self-hosted OpenMemory / Mem0 MCP server

If neither is configured, MockMem0Client is used automatically by
gateway._build_services(). MockMem0Client stores data in-process only
(lost on restart); it is intentionally replaced by real Mem0 in production.

All public methods are async and share the same interface so callers
can swap backends without changes.
"""
import asyncio
import logging

log = logging.getLogger("xiaozhi.src")

# ── Shared result helpers ──────────────────────────────────────────────────────

def _extract_text(hit: dict) -> str:
    """Pull the memory text out of a search or get_all result dict."""
    return hit.get("memory") or hit.get("text") or hit.get("content") or ""


# ── Real Mem0 client ──────────────────────────────────────────────────────────

class Mem0Client:
    """
    Thin async wrapper around the mem0ai SDK.

    Instantiation is lazy: the SDK is imported and the HTTP client is created
    on the first actual call. This keeps startup fast when mem0ai is installed
    but credentials are not yet configured.
    """

    def __init__(self, api_key: str = "", server_url: str = ""):
        if not api_key and not server_url:
            raise ValueError("Provide MEM0_API_KEY or MEM0_SERVER_URL")
        self._api_key = api_key
        self._server_url = server_url
        self._client = None   # lazy init

    def _get(self):
        if self._client is None:
            try:
                from mem0 import MemoryClient
            except ImportError as exc:
                raise RuntimeError(
                    "mem0ai not installed — run: pip install mem0ai"
                ) from exc

            if self._api_key:
                self._client = MemoryClient(api_key=self._api_key)
                log.debug("Mem0Client: cloud mode")
            else:
                self._client = MemoryClient(host=self._server_url)
                log.debug("Mem0Client: self-hosted mode (%s)", self._server_url)
        return self._client

    async def search(
        self, query: str, user_id: str, top_k: int = 5
    ) -> list[dict]:
        """Semantic search. Returns list of {memory, score, ...} dicts."""
        client = self._get()
        results = await asyncio.to_thread(
            client.search, query, user_id=user_id, top_k=top_k
        )
        return results or []

    async def add(self, messages: list[dict], user_id: str) -> None:
        """Store a conversation turn as a memory."""
        client = self._get()
        await asyncio.to_thread(
            client.add,
            messages,
            user_id=user_id,
            output_format="v1.1",   # suppress deprecation warning
        )

    async def get_all(self, user_id: str) -> list[dict]:
        """Retrieve all memories for a user."""
        client = self._get()
        results = await asyncio.to_thread(
            client.get_all, user_id=user_id
        )
        return results or []


# ── In-process fallback ───────────────────────────────────────────────────────

class MockMem0Client:
    """
    Drop-in replacement for Mem0Client when no credentials are set.

    Data is stored in-process (lost on restart). Automatically activated when
    neither MEM0_API_KEY nor MEM0_SERVER_URL is in the environment.

    Replace with real Mem0Client by adding credentials to .env.
    """

    MAX_PER_USER = 100

    def __init__(self) -> None:
        # user_id → list[str]
        self._store: dict[str, list[str]] = {}

    def _entries(self, user_id: str) -> list[str]:
        return self._store.setdefault(user_id, [])

    async def search(
        self, query: str, user_id: str, top_k: int = 5
    ) -> list[dict]:
        """Return the most recent N entries (no real semantic ranking)."""
        entries = self._entries(user_id)
        return [{"memory": e} for e in entries[-top_k:]]

    async def add(self, messages: list[dict], user_id: str) -> None:
        """Store user utterances from the messages list."""
        entries = self._entries(user_id)
        for msg in messages:
            if msg.get("role") == "user" and msg.get("content"):
                entries.append(msg["content"])
        # Bounded
        self._store[user_id] = entries[-self.MAX_PER_USER:]

    async def get_all(self, user_id: str) -> list[dict]:
        entries = self._entries(user_id)
        return [{"memory": e} for e in entries]

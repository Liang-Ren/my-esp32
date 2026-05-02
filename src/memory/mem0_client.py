"""
Mem0 memory backend.

Two production modes (set one in .env):
  MEM0_API_KEY     — Mem0 cloud (https://mem0.ai)  — uses mem0ai SDK
  MEM0_SERVER_URL  — Self-hosted OpenMemory MCP server — uses direct HTTP

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


# ── Cloud Mem0 client (mem0ai SDK) ────────────────────────────────────────────

class Mem0Client:
    """
    Thin async wrapper around the mem0ai SDK for Mem0 cloud.

    Only used when MEM0_API_KEY is set. For self-hosted OpenMemory use
    OpenMemoryClient instead.
    """

    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("Provide MEM0_API_KEY for cloud mode")
        self._api_key = api_key
        self._client = None   # lazy init

    def _get(self):
        if self._client is None:
            try:
                from mem0 import MemoryClient
            except ImportError as exc:
                raise RuntimeError(
                    "mem0ai not installed — run: pip install mem0ai"
                ) from exc
            self._client = MemoryClient(api_key=self._api_key)
            log.debug("Mem0Client: cloud mode")
        return self._client

    async def search(self, query: str, user_id: str, top_k: int = 5) -> list[dict]:
        client = self._get()
        results = await asyncio.to_thread(
            client.search, query, user_id=user_id, top_k=top_k
        )
        return results or []

    async def add(self, messages: list[dict], user_id: str) -> None:
        client = self._get()
        await asyncio.to_thread(
            client.add,
            messages,
            user_id=user_id,
            output_format="v1.1",
        )

    async def get_all(self, user_id: str) -> list[dict]:
        client = self._get()
        results = await asyncio.to_thread(client.get_all, user_id=user_id)
        return results or []


# ── Self-hosted OpenMemory HTTP client ────────────────────────────────────────

class OpenMemoryClient:
    """
    Direct HTTP adapter for self-hosted OpenMemory (mem0/openmemory-mcp).

    Bypasses the mem0ai SDK — which targets /v1/ paths — and calls the
    OpenMemory REST API at /api/v1/ directly.

    Requires: httpx  (already a transitive dep of openai>=1.0)
    """

    def __init__(self, server_url: str) -> None:
        self._base = server_url.rstrip("/")

    def _http(self):
        try:
            import httpx
        except ImportError as exc:
            raise RuntimeError("httpx not installed — run: pip install httpx") from exc
        return httpx.AsyncClient(base_url=self._base, timeout=10.0)

    async def search(self, query: str, user_id: str, top_k: int = 5) -> list[dict]:
        """
        Return the most recent `top_k` memories for the user.

        OpenMemory's REST API uses SQL ILIKE for search_query, which misses
        semantic matches. Returning recent memories lets the LLM decide
        relevance — adequate for small per-user stores (< 50 facts).
        """
        size = min(top_k, 100)
        async with self._http() as client:
            resp = await client.get(
                "/api/v1/memories/",
                params={
                    "user_id": user_id,
                    "size": size,
                    "sort_column": "created_at",
                    "sort_direction": "desc",
                },
            )
            resp.raise_for_status()
            data = resp.json()
            items = data.get("items") or (data if isinstance(data, list) else [])
            return [{"memory": item["content"]} for item in items if item.get("content")]

    async def add(self, messages: list[dict], user_id: str) -> None:
        """
        Store a conversation turn as a memory.

        Concatenates the message list into a single text and posts it with
        infer=true so OpenMemory's LLM extracts discrete facts automatically.
        """
        parts = []
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role and content:
                parts.append(f"{role}: {content}")
        if not parts:
            return
        text = "\n".join(parts)
        async with self._http() as client:
            resp = await client.post(
                "/api/v1/memories/",
                json={"user_id": user_id, "text": text, "infer": True},
            )
            resp.raise_for_status()

    async def get_all(self, user_id: str) -> list[dict]:
        """Retrieve all memories for a user via GET /api/v1/memories/ (paginated, max 100/page)."""
        results: list[dict] = []
        page = 1
        async with self._http() as client:
            while True:
                resp = await client.get(
                    "/api/v1/memories/",
                    params={"user_id": user_id, "size": 100, "page": page},
                )
                resp.raise_for_status()
                data = resp.json()
                items = data.get("items") or (data if isinstance(data, list) else [])
                results.extend(
                    {"memory": item["content"]} for item in items if item.get("content")
                )
                if page >= data.get("pages", 1):
                    break
                page += 1
        return results


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
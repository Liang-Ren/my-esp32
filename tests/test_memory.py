"""
Test 5 — Mem0 failure falls back gracefully to SQLite
Test 8 — Multiple device_id values maintain separate memory
"""
import pytest
from unittest.mock import AsyncMock

from src.memory.memory_service import MemoryService
from src.memory.mem0_client import MockMem0Client

from tests.conftest import FakeSqlite


# ── Test 5: Mem0 failure → graceful SQLite fallback ───────────────────────────

class TestMem0Fallback:
    """When Mem0 raises, MemoryService continues with SQLite facts."""

    @pytest.fixture
    def failing_mem0(self) -> AsyncMock:
        m = AsyncMock()
        m.search.side_effect = RuntimeError("Mem0 connection refused")
        m.add.side_effect = RuntimeError("Mem0 connection refused")
        return m

    async def test_getRelevantMemories_returns_sqlite_facts_on_mem0_failure(
        self, failing_mem0
    ):
        sqlite = FakeSqlite()
        sqlite.update_long_term(
            "device_1", facts=["user lives in Vancouver", "user likes hiking"]
        )
        svc = MemoryService(mem0_client=failing_mem0, sqlite_memory=sqlite)

        memories = await svc.getRelevantMemories("u1", "device_1", "what do I like?")

        assert "user likes hiking" in memories
        assert "user lives in Vancouver" in memories

    async def test_getRelevantMemories_returns_empty_when_both_fail(
        self, failing_mem0
    ):
        sqlite = FakeSqlite()  # no pre-seeded facts
        svc = MemoryService(mem0_client=failing_mem0, sqlite_memory=sqlite)

        memories = await svc.getRelevantMemories("u1", "device_1", "anything")

        assert memories == []

    async def test_addInteraction_returns_false_on_mem0_write_failure(
        self, failing_mem0
    ):
        sqlite = FakeSqlite()
        svc = MemoryService(mem0_client=failing_mem0, sqlite_memory=sqlite)

        result = await svc.addInteraction("u1", "device_1", "hello", "hi")

        # Mem0 write failed → False
        assert result is False

    async def test_sqlite_still_written_even_when_mem0_fails(self, failing_mem0):
        sqlite = FakeSqlite()
        svc = MemoryService(mem0_client=failing_mem0, sqlite_memory=sqlite)

        await svc.addInteraction("u1", "device_1", "hello there", "hi back")

        history = svc.get_history("device_1")
        assert any(m["content"] == "hello there" for m in history)
        assert any(m["content"] == "hi back" for m in history)

    async def test_getUserProfile_succeeds_even_when_mem0_absent(self):
        sqlite = FakeSqlite()
        sqlite.update_long_term(
            "device_1",
            summary="Power user",
            facts=["fact1"],
            preferences={"lang": "zh"},
        )
        # No Mem0 configured at all
        svc = MemoryService(mem0_client=None, sqlite_memory=sqlite)

        profile = await svc.getUserProfile("u1", "device_1")

        assert profile["summary"] == "Power user"
        assert "fact1" in profile["facts"]
        assert "lang:zh" in profile["preferences_str"]

    async def test_mem0_down_does_not_raise_exception(self, failing_mem0):
        """The gateway must never crash due to a Mem0 failure."""
        svc = MemoryService(mem0_client=failing_mem0, sqlite_memory=FakeSqlite())
        # None of these should raise
        await svc.getRelevantMemories("u1", "d1", "test query")
        await svc.addInteraction("u1", "d1", "user says", "ai says")
        await svc.getUserProfile("u1", "d1")
        await svc.updateImportantFacts("u1", "d1", ["fact"])


# ── Test 8: Multiple device_id values stay isolated ───────────────────────────

class TestMultiDeviceIsolation:
    """Memories and history must be scoped per device_id / user_id."""

    async def test_mem0_searches_are_scoped_by_user_id(self):
        """Mem0 receives the correct user_id for each device."""
        received_user_ids: list[str] = []

        async def tracking_search(query, user_id, top_k=5):
            received_user_ids.append(user_id)
            if user_id == "xiaozhi_device_A":
                return [{"memory": "A's memory"}]
            return [{"memory": "B's memory"}]

        mem0 = AsyncMock()
        mem0.search = tracking_search

        svc = MemoryService(mem0_client=mem0, sqlite_memory=FakeSqlite())

        mems_a = await svc.getRelevantMemories("xiaozhi_device_A", "device_A", "q")
        mems_b = await svc.getRelevantMemories("xiaozhi_device_B", "device_B", "q")

        assert "xiaozhi_device_A" in received_user_ids
        assert "xiaozhi_device_B" in received_user_ids
        assert mems_a == ["A's memory"]
        assert mems_b == ["B's memory"]
        assert "A's memory" not in mems_b
        assert "B's memory" not in mems_a

    async def test_sqlite_history_is_scoped_by_device_id(self):
        sqlite = FakeSqlite()
        svc = MemoryService(mem0_client=None, sqlite_memory=sqlite)

        await svc.addInteraction("u_A", "device_A", "A's question", "A's answer")
        await svc.addInteraction("u_B", "device_B", "B's question", "B's answer")

        history_a = svc.get_history("device_A")
        history_b = svc.get_history("device_B")

        contents_a = {m["content"] for m in history_a}
        contents_b = {m["content"] for m in history_b}

        assert "A's question" in contents_a
        assert "A's answer" in contents_a
        assert "B's question" not in contents_a   # no cross-contamination

        assert "B's question" in contents_b
        assert "B's answer" in contents_b
        assert "A's question" not in contents_b

    async def test_sqlite_facts_are_scoped_by_device_id(self):
        sqlite = FakeSqlite()
        sqlite.update_long_term("device_A", facts=["A fact"])
        sqlite.update_long_term("device_B", facts=["B fact"])
        svc = MemoryService(mem0_client=None, sqlite_memory=sqlite)

        profile_a = await svc.getUserProfile("u_A", "device_A")
        profile_b = await svc.getUserProfile("u_B", "device_B")

        assert "A fact" in profile_a["facts"]
        assert "B fact" not in profile_a["facts"]

        assert "B fact" in profile_b["facts"]
        assert "A fact" not in profile_b["facts"]

    async def test_mock_mem0_client_is_scoped_by_user_id(self):
        """MockMem0Client (in-process fallback) must also scope by user_id."""
        mock = MockMem0Client()
        svc = MemoryService(mem0_client=mock, sqlite_memory=FakeSqlite())

        await svc.addInteraction("u_A", "device_A", "A said this", "AI replied A")
        await svc.addInteraction("u_B", "device_B", "B said this", "AI replied B")

        mems_a = await svc.getRelevantMemories("u_A", "device_A", "what did A say?")
        mems_b = await svc.getRelevantMemories("u_B", "device_B", "what did B say?")

        contents_a = set(mems_a)
        contents_b = set(mems_b)

        assert "A said this" in contents_a
        assert "B said this" not in contents_a

        assert "B said this" in contents_b
        assert "A said this" not in contents_b

    async def test_updateImportantFacts_is_scoped_by_device_id(self):
        sqlite = FakeSqlite()
        svc = MemoryService(mem0_client=None, sqlite_memory=sqlite)

        await svc.updateImportantFacts("u_A", "device_A", ["A is a developer"])
        await svc.updateImportantFacts("u_B", "device_B", ["B is a designer"])

        profile_a = await svc.getUserProfile("u_A", "device_A")
        profile_b = await svc.getUserProfile("u_B", "device_B")

        assert "A is a developer" in profile_a["facts"]
        assert "B is a designer" not in profile_a["facts"]

        assert "B is a designer" in profile_b["facts"]
        assert "A is a developer" not in profile_b["facts"]

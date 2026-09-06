import tempfile

import pytest

from infinitum.config import AppConfig
from infinitum.database import Database
from infinitum.embeddings import EmbeddingClient
from infinitum.learning import MemoryLearner
from infinitum.models import Event, Memory, MemoryCandidate
from infinitum.retrieval import MemoryRetriever
from infinitum.upstream import UpstreamClient

# Zero-signal fixture: a high importance/confidence fact whose content AND topic
# share no tokens with the query "pasta sauce". Pre-gate it scored ~0.286 from the
# flat importance/confidence/freshness terms alone and was therefore always present.
ZERO_SIGNAL_CONTENT = (
    "We installed new stainless steel sinks in the kitchen last spring."
)


async def _retriever(tmp: str, db_out: list | None = None):
    cfg = AppConfig()
    cfg.memory.database_path = f"{tmp}/runtime.db"
    db = Database(cfg.memory.database_path)
    await db.connect()
    embeddings = EmbeddingClient(cfg.embeddings)
    retriever = MemoryRetriever(db, embeddings, cfg)
    return cfg, db, embeddings, retriever


async def _seed_zero_signal_fact(db: Database, memory_type: str = "fact", importance: float = 0.95):
    event = await db.add_event(
        Event(
            session_id="s1",
            event_type="message.user",
            role="user",
            content=ZERO_SIGNAL_CONTENT,
        )
    )
    return await db.create_memory(
        Memory(
            memory_type=memory_type,
            topic="kitchen inventory",
            content=ZERO_SIGNAL_CONTENT,
            importance=importance,
            confidence=0.95,
            source_event_ids=[event.id],
        )
    )


@pytest.mark.asyncio
async def test_zero_signal_fact_is_absent_despite_high_importance():
    with tempfile.TemporaryDirectory() as tmp:
        cfg, db, embeddings, retriever = await _retriever(tmp)
        try:
            memory = await _seed_zero_signal_fact(db)
            results = await retriever.search("pasta sauce", limit=10)
            ids = [item.memory.id for item in results]
            # Pre-gate this scored ~0.286 (> the 0.18 floor) and was present.
            assert memory.id not in ids
        finally:
            await embeddings.close()
            await db.close()


@pytest.mark.asyncio
async def test_high_authority_goal_bypasses_relevance_gate():
    with tempfile.TemporaryDirectory() as tmp:
        cfg, db, embeddings, retriever = await _retriever(tmp)
        try:
            memory = await _seed_zero_signal_fact(db, memory_type="goal", importance=0.9)
            results = await retriever.search("pasta sauce", limit=10)
            ids = [item.memory.id for item in results]
            # Same zero-signal record, but goal + importance >= 0.85 is exempt.
            assert memory.id in ids
        finally:
            await embeddings.close()
            await db.close()


@pytest.mark.asyncio
async def test_lexically_matching_fact_still_present_with_gate():
    with tempfile.TemporaryDirectory() as tmp:
        cfg, db, embeddings, retriever = await _retriever(tmp)
        try:
            event = await db.add_event(
                Event(
                    session_id="s1",
                    event_type="message.user",
                    role="user",
                    content="The pasta sauce recipe starts with San Marzano tomatoes.",
                )
            )
            memory = await db.create_memory(
                Memory(
                    memory_type="fact",
                    topic="cooking",
                    content="The pasta sauce recipe starts with San Marzano tomatoes.",
                    importance=0.95,
                    confidence=0.95,
                    source_event_ids=[event.id],
                )
            )
            results = await retriever.search("pasta sauce", limit=10)
            ids = [item.memory.id for item in results]
            # Characterization: genuine relevance passes both before and after.
            assert memory.id in ids
        finally:
            await embeddings.close()
            await db.close()


@pytest.mark.asyncio
async def test_minimum_relevance_score_zero_restores_legacy_behavior():
    with tempfile.TemporaryDirectory() as tmp:
        cfg, db, embeddings, retriever = await _retriever(tmp)
        try:
            cfg.memory.minimum_relevance_score = 0.0
            memory = await _seed_zero_signal_fact(db)
            results = await retriever.search("pasta sauce", limit=10)
            ids = [item.memory.id for item in results]
            # The 0.0 knob disables the gate entirely: exact legacy behavior.
            assert memory.id in ids
        finally:
            await embeddings.close()
            await db.close()


@pytest.mark.asyncio
async def test_gate_does_not_starve_auto_reinforcement():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = AppConfig()
        cfg.memory.database_path = f"{tmp}/runtime.db"
        db = Database(cfg.memory.database_path)
        await db.connect()
        embeddings = EmbeddingClient(cfg.embeddings)
        upstream = UpstreamClient(cfg)
        retriever = MemoryRetriever(db, embeddings, cfg)
        learner = MemoryLearner(db, retriever, embeddings, upstream, cfg)
        try:
            e1 = await db.add_event(
                Event(
                    session_id="s1",
                    event_type="message.user",
                    role="user",
                    content="We keep cast iron skillets as the primary everyday cookware.",
                )
            )
            existing = await db.create_memory(
                Memory(
                    memory_type="fact",
                    topic="kitchen-tools",
                    content="We keep cast iron skillets as the primary everyday cookware.",
                    source_event_ids=[e1.id],
                )
            )
            # The originating interaction barely mentions the memory.
            e2 = await db.add_event(
                Event(
                    session_id="s2",
                    event_type="message.user",
                    role="user",
                    content="Anyway, do you have a good pasta sauce recommendation?",
                )
            )
            # Real retriever (no AsyncMock): the gate sees query=candidate.content,
            # so the near-identical candidate's lexical signal clears it.
            candidate = MemoryCandidate(
                memory_type="fact",
                topic="kitchen-tools",
                content="We keep cast iron skillets as the primary cookware.",
            )
            affected = await learner._apply(candidate, [e2.id], {existing.id})
            assert affected == {existing.id}
            memories = await db.list_memories(limit=10)
            assert len(memories) == 1
            reinforced = await db.get_memory(existing.id)
            assert reinforced is not None
            assert reinforced.observation_count == 2
        finally:
            await upstream.close()
            await embeddings.close()
            await db.close()

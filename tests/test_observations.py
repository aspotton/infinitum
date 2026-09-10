"""First-class memory observations: write paths, idempotency, cascades.

The fingerprint-keyed ``memory_observations`` table makes replayed evidence
structurally idempotent, and ``observation_count`` on memories stays a cached
counter equal to 1 (creation) + genuinely new observation rows since.
"""

import hashlib
import tempfile
from unittest.mock import AsyncMock

from fastapi.testclient import TestClient

from infinitum.app import create_app
from infinitum.config import AppConfig
from infinitum.database import Database
from infinitum.embeddings import EmbeddingClient
from infinitum.learning import MemoryLearner
from infinitum.models import Event, Memory, MemoryCandidate, ScoredMemory
from infinitum.retrieval import MemoryRetriever
from infinitum.upstream import UpstreamClient


async def _db(tmp: str) -> Database:
    db = Database(f"{tmp}/runtime.db")
    await db.connect()
    return db


async def _learner(tmp: str):
    cfg = AppConfig()
    cfg.memory.database_path = f"{tmp}/runtime.db"
    db = Database(cfg.memory.database_path)
    await db.connect()
    embeddings = EmbeddingClient(cfg.embeddings)
    upstream = UpstreamClient(cfg)
    retriever = MemoryRetriever(db, embeddings, cfg)
    learner = MemoryLearner(db, retriever, embeddings, upstream, cfg)
    return cfg, db, learner, retriever


async def _add_event(db: Database, session_id: str, content: str) -> Event:
    event = Event(session_id=session_id, event_type="message.user", role="user", content=content)
    await db.add_event(event)
    return event


async def test_learner_created_memory_gets_one_user_assertion_observation():
    with tempfile.TemporaryDirectory() as tmp:
        cfg, db, learner, retriever = await _learner(tmp)
        try:
            e1 = await _add_event(db, "s1", "We switched our ids to uuid7.")
            e2 = await _add_event(db, "s1", "uuid7 it is, ordered time-based ids.")
            retriever.search = AsyncMock(return_value=[])
            candidate = MemoryCandidate(
                topic="identifiers", content="The project uses uuid7 identifiers.", confidence=0.8
            )
            affected = await learner._apply(candidate, [e1.id, e2.id], set())
            memory = await db.get_memory(next(iter(affected)))
            assert memory is not None
            observations = await db.list_observations(memory.id)
            assert len(observations) == 1
            obs = observations[0]
            assert obs.evidence_type == "user_assertion"
            assert obs.evidence_weight == 1.0
            assert obs.confidence == memory.confidence
            assert obs.observed_at is not None
            assert sorted(obs.source_event_ids) == sorted([e1.id, e2.id])
            assert memory.observation_count == 1
        finally:
            await db.close()


async def test_manual_post_memory_gets_manual_admin_observation():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = AppConfig()
        cfg.memory.database_path = f"{tmp}/runtime.db"
        cfg.learning.enabled = False
        app = create_app(cfg)
        with TestClient(app) as client:
            response = client.post("/memory", json={"content": "Staging runs in eu-central-1."})
            assert response.status_code == 200
            memory_id = response.json()["id"]
            observations = await app.state.runtime.db.list_observations(memory_id)
        assert len(observations) == 1
        assert observations[0].evidence_type == "manual_admin"
        assert observations[0].evidence_weight == 1.0


async def test_two_distinct_interactions_reinforce_add_observations_and_count():
    with tempfile.TemporaryDirectory() as tmp:
        db = await _db(tmp)
        try:
            e1 = await _add_event(db, "s1", "we use postgres for the primary db")
            e2 = await _add_event(db, "s2", "postgres is still our primary db")
            e3 = await _add_event(db, "s3", "confirmed postgres is the primary db")
            memory = await db.create_memory(
                Memory(topic="database", content="we use postgres", source_event_ids=[e1.id])
            )
            await db.reinforce_memory(
                memory.id, confidence=0.8, importance=0.6, source_event_ids=[e2.id]
            )
            await db.reinforce_memory(
                memory.id, confidence=0.8, importance=0.6, source_event_ids=[e3.id]
            )
            updated = await db.get_memory(memory.id)
            assert updated is not None
            assert updated.observation_count == 3
            observations = await db.list_observations(memory.id)
            assert len(observations) == 3
            linked = [sorted(obs.source_event_ids) for obs in observations]
            assert sorted([e1.id]) in linked
            assert sorted([e2.id]) in linked
            assert sorted([e3.id]) in linked
        finally:
            await db.close()


async def test_replaying_the_same_learn_job_twice_is_idempotent():
    with tempfile.TemporaryDirectory() as tmp:
        cfg, db, learner, retriever = await _learner(tmp)
        try:
            e1 = await _add_event(db, "s1", "We standardized on PG17 for relational persistence.")
            e2 = await _add_event(db, "s2", "PG17 remains our relational standard.")
            existing = await db.create_memory(
                Memory(
                    memory_type="decision",
                    topic="database",
                    content="We standardized on PG17 for relational persistence.",
                    source_event_ids=[e1.id],
                )
            )
            retriever.search = AsyncMock(
                return_value=[
                    ScoredMemory(
                        memory=existing,
                        score=0.78,
                        semantic_score=0.95,
                        lexical_score=0.20,
                        topic_score=0.8,
                        freshness_score=1.0,
                    )
                ]
            )
            candidate = MemoryCandidate(
                memory_type="decision",
                topic="database",
                content="We standardized on PG17 for relational persistence.",
                confidence=0.9,
            )
            await learner._apply(candidate, [e2.id], {existing.id})
            assert len(await db.list_observations(existing.id)) == 2
            first = await db.get_memory(existing.id)
            assert first is not None and first.observation_count == 2

            # Replay the same job: identical candidate, identical source events.
            await learner._apply(candidate, [e2.id], {existing.id})
            assert len(await db.list_observations(existing.id)) == 2
            replayed = await db.get_memory(existing.id)
            assert replayed is not None and replayed.observation_count == 2
        finally:
            await db.close()


async def test_hard_delete_cascades_observations_and_sources():
    with tempfile.TemporaryDirectory() as tmp:
        db = await _db(tmp)
        try:
            e1 = await _add_event(db, "s1", "temporary fact with provenance")
            memory = await db.create_memory(
                Memory(topic="tmp", content="temporary fact", source_event_ids=[e1.id])
            )
            assert len(await db.list_observations(memory.id)) == 1
            await db.execute("DELETE FROM memories WHERE id=?", (memory.id,))
            total = await db.fetchone("SELECT COUNT(*) AS n FROM memory_observations")
            assert int(total["n"]) == 0
            sources = await db.fetchone("SELECT COUNT(*) AS n FROM memory_observation_sources")
            assert int(sources["n"]) == 0
        finally:
            await db.close()


async def test_create_with_empty_source_ids_still_records_observation():
    with tempfile.TemporaryDirectory() as tmp:
        db = await _db(tmp)
        try:
            memory = await db.create_memory(Memory(topic="t", content="no provenance events"))
            observations = await db.list_observations(memory.id)
            assert len(observations) == 1
            assert observations[0].source_event_ids == []
            assert observations[0].fingerprint
        finally:
            await db.close()


async def test_reinitialize_adds_observation_tables_to_old_database():
    with tempfile.TemporaryDirectory() as tmp:
        db = await _db(tmp)
        try:
            # Simulate a pre-migration database: drop the new tables, re-init.
            await db.executescript(
                "DROP TABLE memory_observation_sources; DROP TABLE memory_observations;"
            )
            await db.initialize()
            tables = {
                r["name"]
                for r in await db.fetchall("SELECT name FROM sqlite_master WHERE type='table'")
            }
            assert {"memory_observations", "memory_observation_sources"} <= tables
            indexes = {
                r["name"]
                for r in await db.fetchall("SELECT name FROM sqlite_master WHERE type='index'")
            }
            assert "idx_memory_observations_memory" in indexes
        finally:
            await db.close()


# Baseline characterization of the GET /memory LIST response, captured from
# the unchanged code (serialized Memory fields). The detail endpoint may only
# ADD the observations key; this list shape must never drift.
_LIST_MEMORY_KEYS = [
    "id",
    "memory_type",
    "topic",
    "content",
    "status",
    "importance",
    "confidence",
    "observation_count",
    "created_at",
    "updated_at",
    "last_accessed_at",
    "superseded_by",
    "metadata",
    "source_event_ids",
    "valid_from",
    "valid_until",
    "observed_at",
]


async def test_backfill_adds_one_legacy_observation_per_memory_without_touching_counts():
    with tempfile.TemporaryDirectory() as tmp:
        db = await _db(tmp)
        try:
            e1 = await _add_event(db, "s1", "primary db decision evidence")
            e2 = await _add_event(db, "s2", "staging region evidence")
            reinforced = await db.create_memory(
                Memory(topic="database", content="we use postgres", source_event_ids=[e1.id])
            )
            with_sources = await db.create_memory(
                Memory(
                    topic="regions",
                    content="staging runs in eu-central-1",
                    source_event_ids=[e2.id],
                )
            )
            no_sources = await db.create_memory(Memory(topic="misc", content="orphan fact"))
            # Legacy counter state a pre-migration database would carry: the
            # backfill must preserve it byte-identically, never recompute.
            await db.execute("UPDATE memories SET observation_count=5 WHERE id=?", (reinforced.id,))
            # Emulate a pre-observations database: these tables did not exist.
            await db.executescript(
                "DROP TABLE memory_observation_sources; DROP TABLE memory_observations;"
            )
            await db.initialize()

            rows = await db.fetchall("SELECT * FROM memory_observations")
            assert len(rows) == 3
            assert {r["evidence_type"] for r in rows} == {"legacy"}
            by_id = {r["memory_id"]: r for r in rows}
            assert set(by_id) == {reinforced.id, with_sources.id, no_sources.id}
            expected_counts = ((reinforced.id, 5), (with_sources.id, 1), (no_sources.id, 1))
            for mid, expected_count in expected_counts:
                memory = await db.get_memory(mid)
                assert memory is not None
                assert memory.observation_count == expected_count
                obs = await db.list_observations(mid)
                assert len(obs) == 1
                assert obs[0].evidence_type == "legacy"
                # 'legacy' is not in the full-weight evidence set -> 0.5.
                assert obs[0].evidence_weight == 0.5
                assert obs[0].observed_at == memory.created_at
                assert obs[0].confidence == memory.confidence
                assert by_id[mid]["confidence"] == memory.confidence
                legacy_fp = hashlib.sha256(f"legacy|{mid}".encode()).hexdigest()
                assert obs[0].fingerprint == legacy_fp
            mirrored = await db.list_observations(with_sources.id)
            assert mirrored[0].source_event_ids == [e2.id]
            orphan = await db.list_observations(no_sources.id)
            assert orphan[0].source_event_ids == []
        finally:
            await db.close()


async def test_second_initialize_after_backfill_adds_nothing():
    with tempfile.TemporaryDirectory() as tmp:
        db = await _db(tmp)
        try:
            memory = await db.create_memory(Memory(topic="t", content="a legacy-era fact"))
            await db.executescript(
                "DROP TABLE memory_observation_sources; DROP TABLE memory_observations;"
            )
            await db.initialize()
            first = sorted(
                (r["id"], r["fingerprint"])
                for r in await db.fetchall("SELECT id, fingerprint FROM memory_observations")
            )
            await db.initialize()
            second = sorted(
                (r["id"], r["fingerprint"])
                for r in await db.fetchall("SELECT id, fingerprint FROM memory_observations")
            )
            assert second == first
            assert len(first) == 1
            assert (await db.get_memory(memory.id)).observation_count == 1
        finally:
            await db.close()


async def test_backfill_is_a_no_op_on_a_current_schema_database():
    with tempfile.TemporaryDirectory() as tmp:
        db = await _db(tmp)
        try:
            memory = await db.create_memory(Memory(topic="t", content="born after the migration"))
            await db.initialize()
            legacy = await db.fetchone(
                "SELECT COUNT(*) AS n FROM memory_observations WHERE evidence_type='legacy'"
            )
            assert int(legacy["n"]) == 0
            observations = await db.list_observations(memory.id)
            assert len(observations) == 1
            assert observations[0].evidence_type == "user_assertion"
        finally:
            await db.close()


async def test_get_memory_detail_surfaces_observations_and_list_shape_stays_identical():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = AppConfig()
        cfg.memory.database_path = f"{tmp}/runtime.db"
        cfg.learning.enabled = False
        app = create_app(cfg)
        with TestClient(app) as client:
            response = client.post("/memory", json={"content": "Detail views show evidence."})
            assert response.status_code == 200
            memory_id = response.json()["id"]
            detail = client.get(f"/memory/{memory_id}")
            assert detail.status_code == 200
            body = detail.json()
            assert sorted(body.keys()) == sorted([*_LIST_MEMORY_KEYS, "observations"])
            assert [obs["evidence_type"] for obs in body["observations"]] == ["manual_admin"]
            listing = client.get("/memory").json()
            assert "observations" not in listing[0]
            assert sorted(listing[0].keys()) == sorted(_LIST_MEMORY_KEYS)

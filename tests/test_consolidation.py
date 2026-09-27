"""MemoryConsolidator: one LLM-proposed, code-applied merge pass per topic.

Covers the periodic-consolidation acceptance matrix: happy merge, churn
no-op, per-cluster conflict recording, loud empty-output raise, parser
fallback, replay idempotence, survivor/race guards, runtime wiring, and
oversized-topic rotation via explicit continuation passes.
"""

import json
import tempfile
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock

import pytest

from infinitum.config import AppConfig
from infinitum.consolidation import MemoryConsolidator
from infinitum.database import Database, _CONSOLIDATION_META_PREFIX
from infinitum.models import Event, Memory
from infinitum.runtime import build_runtime
from infinitum.upstream import UpstreamClient

NEAR_A1 = "The team decided to use SQLite for persistence"
NEAR_A2 = "The team decided to use SQLite for persistence layer"
NEAR_B1 = "The analytics pipeline runs on the spark cluster every night"
NEAR_B2 = "The analytics pipeline runs on the spark cluster every night at midnight"


def _setup(tmp: str, **learning_overrides) -> tuple[AppConfig, Database, UpstreamClient]:
    cfg = AppConfig()
    cfg.memory.database_path = f"{tmp}/runtime.db"
    cfg.embeddings.enabled = False
    for key, value in learning_overrides.items():
        setattr(cfg.learning, key, value)
    return cfg, Database(cfg.memory.database_path), UpstreamClient(cfg)


def _content_body(text: str) -> dict:
    return {
        "choices": [
            {"message": {"role": "assistant", "content": text}, "finish_reason": "stop"}
        ]
    }


def _proposal_body(entries: list[dict]) -> dict:
    return _content_body(json.dumps({"clusters": entries}))


def _tool_call_body(proposal: dict) -> dict:
    return {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "c1",
                            "type": "function",
                            "function": {"name": "emit", "arguments": json.dumps(proposal)},
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ]
    }


def _generic_proposal_fn():
    async def propose(clusters, *, topic, model):
        return {
            "clusters": [
                {
                    "cluster_id": index,
                    "survivor_id": cluster[0].id,
                    "canonical_content": f"canonical {topic} {index}",
                    "conflict": False,
                }
                for index, cluster in enumerate(clusters)
            ]
        }

    return propose


async def _seed(
    db: Database, topic: str, content: str, *, at: datetime | None = None
) -> Memory:
    event = Event(session_id="s", event_type="message.user", role="user", content=content)
    await db.add_event(event)
    memory = Memory(topic=topic, content=content, source_event_ids=[event.id])
    await db.create_memory(memory)
    if at is not None:
        stamp = at.isoformat()
        await db.execute(
            "UPDATE memories SET updated_at=?, created_at=? WHERE id=?",
            (stamp, stamp, memory.id),
        )
        memory.updated_at = at
        memory.created_at = at
    return memory


async def _events_by_type(db: Database, event_type: str) -> list[Event]:
    return [e for e in await db.list_events(limit=200) if e.event_type == event_type]


async def test_merge_happy_path_uses_exactly_one_upstream_call():
    with tempfile.TemporaryDirectory() as tmp:
        cfg, db, upstream = _setup(tmp, consolidation_min_changed_memories=1)
        await db.connect()
        a = await _seed(db, "sqlite-storage", NEAR_A1)
        b = await _seed(db, "sqlite-storage", NEAR_A2)
        upstream.learning_chat_completion = AsyncMock(
            return_value=_proposal_body(
                [
                    {
                        "cluster_id": 0,
                        "survivor_id": a.id,
                        "canonical_content": "The team uses SQLite for persistence.",
                        "conflict": False,
                    }
                ]
            )
        )
        consolidator = MemoryConsolidator(db, upstream, cfg)
        member_observations_before = [
            (o.id, o.fingerprint) for o in await db.list_observations(b.id)
        ]

        stats = await consolidator.consolidate_topic("sqlite-storage", "test-model")

        assert stats == {
            "changed": 2,
            "clusters": 1,
            "merged": 1,
            "conflicts": 0,
            "skipped": 0,
            "no_op": False,
        }
        assert upstream.learning_chat_completion.await_count == 1
        survivor = await db.get_memory(a.id)
        superseded = await db.get_memory(b.id)
        assert survivor.status == "active" and survivor.superseded_by is None
        assert survivor.content == "The team uses SQLite for persistence."
        assert survivor.source_event_ids == sorted([*a.source_event_ids, *b.source_event_ids])
        assert survivor.observation_count == 2
        assert survivor.metadata["last_reinforcement"]["method"] == "consolidation"
        assert len(await db.list_observations(a.id)) == 2  # exactly one new survivor row
        assert superseded.status == "superseded"
        assert superseded.superseded_by == survivor.id
        assert superseded.valid_until is not None
        member_observations_after = [
            (o.id, o.fingerprint) for o in await db.list_observations(b.id)
        ]
        assert member_observations_after == member_observations_before  # evidence untouched

        passes = await _events_by_type(db, "consolidation.pass")
        assert len(passes) == 1 and passes[0].session_id == "consolidation"
        assert json.loads(passes[0].content)["merged"] == 1
        assert await db.get_meta_value(_CONSOLIDATION_META_PREFIX + "sqlite-storage")
        dirty_ids = [mid for mid, _at in await db.get_topic_updates("sqlite-storage")]
        assert survivor.id in dirty_ids


async def test_churn_below_floor_is_a_no_op():
    with tempfile.TemporaryDirectory() as tmp:
        cfg, db, upstream = _setup(tmp)  # default churn floor = 8
        await db.connect()
        await _seed(db, "t", NEAR_A1)
        await _seed(db, "t", NEAR_A2)
        upstream.learning_chat_completion = AsyncMock(return_value=_content_body("{}"))
        consolidator = MemoryConsolidator(db, upstream, cfg)

        stats = await consolidator.consolidate_topic("t", "test-model")

        assert stats["no_op"] is True and stats["merged"] == 0 and stats["changed"] == 2
        assert upstream.learning_chat_completion.await_count == 0
        assert len(await db.list_active_topic_memories("t")) == 2  # zero mutations
        passes = await _events_by_type(db, "consolidation.pass")
        assert len(passes) == 1 and json.loads(passes[0].content)["no_op"] is True
        assert await db.get_meta_value(_CONSOLIDATION_META_PREFIX + "t")


async def test_conflict_cluster_is_recorded_and_other_clusters_still_apply():
    with tempfile.TemporaryDirectory() as tmp:
        cfg, db, upstream = _setup(tmp, consolidation_min_changed_memories=1)
        await db.connect()
        for content in (NEAR_A1, NEAR_A2, NEAR_B1, NEAR_B2):
            await _seed(db, "mixed", content)
        captured: dict = {}

        async def propose(clusters, *, topic, model):
            captured["clusters"] = clusters
            return {
                "clusters": [
                    {
                        "cluster_id": index,
                        "survivor_id": cluster[0].id,
                        "canonical_content": f"canonical {index}",
                        "conflict": index == 0,
                    }
                    for index, cluster in enumerate(clusters)
                ]
            }

        consolidator = MemoryConsolidator(db, upstream, cfg, proposal_fn=propose)
        stats = await consolidator.consolidate_topic("mixed", "test-model")

        assert stats["clusters"] == 2
        assert stats["conflicts"] == 1 and stats["merged"] == 1
        conflicts = await _events_by_type(db, "consolidation.conflict")
        assert len(conflicts) == 1 and conflicts[0].role == "system"
        assert conflicts[0].metadata["topic"] == "mixed"
        conflicted, applied = captured["clusters"]
        for memory in conflicted:
            current = await db.get_memory(memory.id)
            assert current.status == "active" and current.superseded_by is None
        survivor = await db.get_memory(applied[0].id)
        merged_member = await db.get_memory(applied[1].id)
        assert survivor.status == "active" and survivor.content == "canonical 1"
        assert merged_member.status == "superseded"
        assert merged_member.superseded_by == survivor.id


async def test_empty_or_unparseable_output_raises_before_any_event_or_checkpoint():
    for body in (_content_body(""), _content_body("I decline to merge these.")):
        with tempfile.TemporaryDirectory() as tmp:
            cfg, db, upstream = _setup(tmp, consolidation_min_changed_memories=1)
            await db.connect()
            await _seed(db, "t", NEAR_A1)
            await _seed(db, "t", NEAR_A2)
            upstream.learning_chat_completion = AsyncMock(return_value=body)
            consolidator = MemoryConsolidator(db, upstream, cfg)

            with pytest.raises(RuntimeError):
                await consolidator.consolidate_topic("t", "test-model")

            assert not await _events_by_type(db, "consolidation.pass")
            assert await db.get_meta_value(_CONSOLIDATION_META_PREFIX + "t") is None
            assert len(await db.list_active_topic_memories("t")) == 2


async def test_schema_shaped_tool_call_arguments_are_accepted():
    with tempfile.TemporaryDirectory() as tmp:
        cfg, db, upstream = _setup(tmp, consolidation_min_changed_memories=1)
        await db.connect()
        a = await _seed(db, "t", NEAR_A1)
        b = await db.get_memory((await _seed(db, "t", NEAR_A2)).id)
        upstream.learning_chat_completion = AsyncMock(
            return_value=_tool_call_body(
                {
                    "clusters": [
                        {
                            "cluster_id": 0,
                            "survivor_id": a.id,
                            "canonical_content": "The team uses SQLite for persistence.",
                            "conflict": False,
                        }
                    ]
                }
            )
        )
        consolidator = MemoryConsolidator(db, upstream, cfg)

        stats = await consolidator.consolidate_topic("t", "test-model")

        assert stats["merged"] == 1
        assert (await db.get_memory(b.id)).status == "superseded"


async def test_replay_does_not_inflate_observations():
    with tempfile.TemporaryDirectory() as tmp:
        cfg, db, upstream = _setup(tmp, consolidation_min_changed_memories=1)
        await db.connect()
        a = await _seed(db, "t", NEAR_A1)
        await _seed(db, "t", NEAR_A2)
        upstream.learning_chat_completion = AsyncMock(
            return_value=_proposal_body(
                [
                    {
                        "cluster_id": 0,
                        "survivor_id": a.id,
                        "canonical_content": "The team uses SQLite for persistence.",
                        "conflict": False,
                    }
                ]
            )
        )
        consolidator = MemoryConsolidator(db, upstream, cfg)

        first = await consolidator.consolidate_topic("t", "test-model")
        observations_after_first = len(await db.list_observations(a.id))
        second = await consolidator.consolidate_topic("t", "test-model")

        assert first["merged"] == 1
        assert second["no_op"] is True and second["changed"] == 0
        survivor = await db.get_memory(a.id)
        assert survivor.observation_count == 2
        assert len(await db.list_observations(a.id)) == observations_after_first
        assert upstream.learning_chat_completion.await_count == 1  # pass 2 never reached the LLM


async def test_proposal_naming_a_non_member_survivor_is_skipped():
    with tempfile.TemporaryDirectory() as tmp:
        cfg, db, upstream = _setup(tmp, consolidation_min_changed_memories=1)
        await db.connect()
        await _seed(db, "t", NEAR_A1)
        await _seed(db, "t", NEAR_A2)

        async def propose(clusters, *, topic, model):
            return {
                "clusters": [
                    {
                        "cluster_id": 0,
                        "survivor_id": "mem_does_not_exist",
                        "canonical_content": "nope",
                        "conflict": False,
                    }
                ]
            }

        consolidator = MemoryConsolidator(db, upstream, cfg, proposal_fn=propose)
        stats = await consolidator.consolidate_topic("t", "test-model")

        assert stats["merged"] == 0 and stats["skipped"] == 1
        assert len(await db.list_active_topic_memories("t")) == 2
        assert len(await _events_by_type(db, "consolidation.pass")) == 1


async def test_race_guard_skips_a_member_mutated_mid_pass():
    with tempfile.TemporaryDirectory() as tmp:
        cfg, db, upstream = _setup(tmp, consolidation_min_changed_memories=1)
        await db.connect()
        survivor = await _seed(db, "t", NEAR_A1)
        stale_member = await _seed(db, "t", NEAR_A2)

        async def racy_proposal(clusters, *, topic, model):
            # A foreground learner reinforces the member AFTER the snapshot but
            # BEFORE apply — the stale-evidence guard must leave it active.
            intruder = Event(
                session_id="s2", event_type="message.user", role="user", content="new fact"
            )
            await db.add_event(intruder)
            await db.reinforce_memory(
                stale_member.id,
                confidence=0.8,
                importance=0.6,
                source_event_ids=[intruder.id],
            )
            return {
                "clusters": [
                    {
                        "cluster_id": 0,
                        "survivor_id": survivor.id,
                        "canonical_content": "merged canonical",
                        "conflict": False,
                    }
                ]
            }

        consolidator = MemoryConsolidator(db, upstream, cfg, proposal_fn=racy_proposal)
        stats = await consolidator.consolidate_topic("t", "test-model")

        assert stats["skipped"] == 1 and stats["merged"] == 1
        mutated = await db.get_memory(stale_member.id)
        assert mutated.status == "active" and mutated.superseded_by is None
        assert mutated.content == NEAR_A2
        merged = await db.get_memory(survivor.id)
        assert merged.status == "active" and merged.content == "merged canonical"


async def test_oversized_topic_rotates_via_continuation_pass():
    with tempfile.TemporaryDirectory() as tmp:
        cfg, db, upstream = _setup(
            tmp,
            consolidation_min_changed_memories=1,
            consolidation_max_memories_per_pass=3,
        )
        await db.connect()
        base = datetime.now(timezone.utc) - timedelta(hours=6)
        pair_in_1 = await _seed(db, "rotation-topic", NEAR_A1, at=base)
        pair_in_2 = await _seed(db, "rotation-topic", NEAR_A2, at=base + timedelta(minutes=1))
        await _seed(
            db,
            "rotation-topic",
            "The reporting dashboard refreshes each morning",
            at=base + timedelta(minutes=2),
        )
        pair_out_1 = await _seed(db, "rotation-topic", NEAR_B1, at=base + timedelta(minutes=3))
        pair_out_2 = await _seed(db, "rotation-topic", NEAR_B2, at=base + timedelta(minutes=4))
        consolidator = MemoryConsolidator(
            db, upstream, cfg, proposal_fn=_generic_proposal_fn()
        )

        first = await consolidator.consolidate_topic("rotation-topic", "test-model")

        assert first["changed"] == 5 and first["merged"] == 1
        in_one = await db.get_memory(pair_in_1.id)
        in_two = await db.get_memory(pair_in_2.id)
        assert {in_one.status, in_two.status} == {"active", "superseded"}
        loser_in = in_two if in_one.status == "active" else in_one
        assert loser_in.superseded_by == (in_one if loser_in is in_two else in_two).id
        assert (await db.get_memory(pair_out_1.id)).status == "active"
        assert (await db.get_memory(pair_out_2.id)).status == "active"
        job_row = await db.fetchone(
            "SELECT payload_json FROM jobs WHERE job_type='consolidate_topic'"
            " AND status='pending'"
        )
        assert job_row is not None
        payload = json.loads(job_row["payload_json"])
        assert payload["topic"] == "rotation-topic" and payload["continuation"] is True

        second = await consolidator.consolidate_topic(
            "rotation-topic", "test-model", continuation=True
        )

        assert second["changed"] == 0 and second["no_op"] is False
        assert second["merged"] == 1  # the outside-window pair merged: rotation is real
        out_one = await db.get_memory(pair_out_1.id)
        out_two = await db.get_memory(pair_out_2.id)
        assert {out_one.status, out_two.status} == {"active", "superseded"}


async def test_runtime_wires_the_consolidator_only_when_gated_on():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = AppConfig()
        cfg.memory.database_path = f"{tmp}/runtime.db"
        runtime = await build_runtime(cfg)
        assert runtime.worker.consolidator is None

        cfg2 = AppConfig()
        cfg2.memory.database_path = f"{tmp}/runtime2.db"
        cfg2.learning.consolidation = True
        runtime2 = await build_runtime(cfg2)
        assert isinstance(runtime2.worker.consolidator, MemoryConsolidator)

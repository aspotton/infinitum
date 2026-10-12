import tempfile

import pytest

from infinitum.database import Database
from infinitum.models import Event, Memory


async def _seed(
    db: Database,
    *,
    memory_type: str = "fact",
    topic: str = "general",
    content: str = "plain note",
    importance: float = 0.5,
) -> str:
    memory = await db.create_memory(
        Memory(
            memory_type=memory_type,
            topic=topic,
            content=content,
            importance=importance,
        )
    )
    return memory.id


async def _seed_eight(db: Database) -> dict[str, str]:
    """8-row store touching every arm.

    qnorm is "alpha beta" (LENGTH 9, so the 4x window is 24 non-whitespace
    chars). Rows named miss* are designed to clear no arm.
    """
    return {
        # arm (a): FTS token hit, long topic + content beyond both windows
        "fts": await _seed(
            db,
            content="kappa token appears here in this stored fact about kernels",
            topic="kernel-maintenance-archive-subsystem-long",
        ),
        # arm (c): high-authority goal, long content + topic out of windows
        "goal": await _seed(
            db,
            memory_type="goal",
            importance=0.9,
            content="quarterly objective to finalize the migration of the legacy billing stack",
            topic="billing-migration-program-tracker-x",
        ),
        # arm (c): high-authority decision (exactly the 0.85 boundary)
        "decision": await _seed(
            db,
            memory_type="decision",
            importance=0.85,
            content="the chosen database engine for the analytics pipeline was finalized",
            topic="analytics-engine-selection-committee",
        ),
        # arm (d): content inside the 4x window (20 stripped chars <= 24)
        "window": await _seed(
            db,
            content="tabs\t\there  padded",
            topic="deployment-checklist-for-the-eu-region-board",
        ),
        # arm (e): topic inside the 4x window (18 stripped chars <= 24)
        "topic": await _seed(
            db,
            content="long enough content to stay out of the content window by a margin",
            topic="redis-cache",
        ),
        # arm (d) via the >8192 lexical-opacity OR sub-arm
        "huge": await _seed(
            db,
            content="q" * 9000,
            topic="obscure-overflow-vault-of-long-form-prose-z",
        ),
        # miss: low-importance fact, both fields outside every window, no token hit
        "miss": await _seed(
            db,
            content="lonely fact whose wording rhymes with nothing in the query at all here",
            topic="vault-of-unrelated-marginalia-volume-nine",
        ),
        # miss: ordinary fact below the authority floor and past both windows
        "miss2": await _seed(
            db,
            memory_type="preference",
            importance=0.2,
            content="preference written out long enough to escape the phrase window entirely",
            topic="miscellaneous-surface-preferences-panel-set",
        ),
    }


@pytest.mark.asyncio
async def test_build_retrieval_candidates_returns_exact_superset_over_all_arms():
    with tempfile.TemporaryDirectory() as tmp:
        db = Database(f"{tmp}/runtime.db")
        await db.connect()
        ids = await _seed_eight(db)
        result = await db.build_retrieval_candidates(
            terms=["kappa"], qnorm="alpha beta", limit=50
        )
        await db.close()
        assert result is not None
        expected = {ids["fts"], ids["goal"], ids["decision"], ids["window"], ids["topic"], ids["huge"]}
        # EXACT set equality, not subset: the misses must stay out.
        assert set(result) == expected
        assert len(result) == len(set(result))


@pytest.mark.asyncio
async def test_extra_predicate_sql_drops_goal_row_from_every_arm():
    with tempfile.TemporaryDirectory() as tmp:
        db = Database(f"{tmp}/runtime.db")
        await db.connect()
        ids = await _seed_eight(db)
        result = await db.build_retrieval_candidates(
            terms=["kappa"],
            qnorm="alpha beta",
            limit=50,
            extra_predicate_sql="memory_type != 'goal'",
        )
        await db.close()
        assert result is not None
        assert ids["goal"] not in result
        assert set(result) == {
            ids["fts"], ids["decision"], ids["window"], ids["topic"], ids["huge"]
        }


@pytest.mark.asyncio
async def test_fts_disabled_returns_none_signal_for_full_scan():
    with tempfile.TemporaryDirectory() as tmp:
        db = Database(f"{tmp}/runtime.db")
        await db.connect()
        await _seed_eight(db)
        db.fts_enabled = False
        result = await db.build_retrieval_candidates(
            terms=["kappa"], qnorm="alpha beta", limit=50
        )
        await db.close()
        assert result is None


@pytest.mark.asyncio
async def test_arm_row_count_hitting_limit_is_degenerate_none():
    with tempfile.TemporaryDirectory() as tmp:
        db = Database(f"{tmp}/runtime.db")
        await db.connect()
        for i in range(5):
            await _seed(
                db,
                content=f"note {i}",
                topic="window-filling-store-for-degeneracy-check",
            )
        # Window = 4 * LENGTH("alpha beta gamma") = 76 >= every content length,
        # so arm (d) alone returns exactly limit=5 rows -> degeneracy.
        result = await db.build_retrieval_candidates(
            terms=["none-of-these"], qnorm="alpha beta gamma", limit=5
        )
        await db.close()
        assert result is None


@pytest.mark.asyncio
async def test_huge_content_row_is_a_candidate_regardless_of_qnorm_window():
    with tempfile.TemporaryDirectory() as tmp:
        db = Database(f"{tmp}/runtime.db")
        await db.connect()
        ids = await _seed_eight(db)
        # qnorm length 1 -> window is 4 stripped chars: only the >8192 OR
        # sub-arm can reach the huge row.
        result = await db.build_retrieval_candidates(
            terms=["zq"], qnorm="zq", limit=50
        )
        await db.close()
        assert result is not None
        assert ids["huge"] in result


@pytest.mark.asyncio
async def test_list_memories_by_ids_active_excludes_archived_and_equals_loader_output():
    with tempfile.TemporaryDirectory() as tmp:
        db = Database(f"{tmp}/runtime.db")
        await db.connect()
        # Sources exercise the batched loader's ORDER BY event_id parity.
        ev_b = await db.add_event(
            Event(session_id="s", event_type="request.received", content="b")
        )
        ev_a = await db.add_event(
            Event(session_id="s", event_type="request.received", content="a")
        )
        active = await db.create_memory(Memory(topic="parity", content="active memory"))
        await db.add_memory_source(active.id, ev_b.id)
        await db.add_memory_source(active.id, ev_a.id)
        archived = await db.create_memory(Memory(topic="parity", content="old memory"))
        await db.archive_memory(archived.id)

        result = await db.list_memories_by_ids_active([active.id, archived.id])
        assert [m.id for m in result] == [active.id]
        # Dataclass-equality proof: the batched path returns Memory objects
        # equal to the paged list_memories path, sources ordered by event_id.
        mem2 = result[0]
        mem = next(m for m in await db.list_memories(limit=10) if m.id == active.id)
        assert mem == mem2
        assert mem2.source_event_ids == sorted([ev_a.id, ev_b.id])
        assert await db.list_memories_by_ids_active([]) == []
        await db.close()

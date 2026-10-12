import tempfile
from unittest.mock import AsyncMock

import numpy as np
import pytest

from infinitum.config import AppConfig
from infinitum.database import Database
from infinitum.learning import embedding_backfill_sweep
from infinitum.models import Memory

_VECTOR = np.ones(4, dtype=np.float32)


def _cursor_key(cfg: AppConfig) -> str:
    return f"vector_backfill_cursor:{cfg.embeddings.model}"


async def _seed_active(db: Database, count: int) -> list[str]:
    ids = []
    for i in range(count):
        memory = await db.create_memory(
            Memory(topic=f"backfill-{i}", content=f"durable memory number {i}")
        )
        ids.append(memory.id)
    return ids


async def _embedded_ids(db: Database, model: str) -> set[str]:
    rows = await db.fetchall(
        "SELECT memory_id FROM memory_embeddings WHERE model=?", (model,)
    )
    return {row["memory_id"] for row in rows}


@pytest.mark.asyncio
async def test_sweep_embeds_vectorless_rows_advances_cursor_to_max_id():
    with tempfile.TemporaryDirectory() as tmp:
        db = Database(f"{tmp}/runtime.db")
        await db.connect()
        cfg = AppConfig()
        cfg.embeddings.enabled = True
        embeddings = AsyncMock()
        embeddings.embed = AsyncMock(return_value=_VECTOR)
        ids = await _seed_active(db, 20)
        # Half already vectorized under the configured model: only the other
        # 10 may be embedded, so embed-call counts prove the LEFT JOIN works.
        for memory_id in ids[:10]:
            await db.set_embedding(memory_id, cfg.embeddings.model, _VECTOR)

        await embedding_backfill_sweep(db, embeddings, cfg)

        assert await _embedded_ids(db, cfg.embeddings.model) == set(ids)
        assert embeddings.embed.call_count == 10
        # Cursor = highest id that passed the loop = max of the vectorless
        # half; pre-set ids above it were never selected, so the cursor must
        # sit below them (ids are random and interleave between the halves).
        cursor = await db.get_meta_value(_cursor_key(cfg))
        assert cursor == max(ids[10:])
        await db.close()


@pytest.mark.asyncio
async def test_second_sweep_over_healed_store_makes_zero_embed_calls():
    with tempfile.TemporaryDirectory() as tmp:
        db = Database(f"{tmp}/runtime.db")
        await db.connect()
        cfg = AppConfig()
        cfg.embeddings.enabled = True
        embeddings = AsyncMock()
        embeddings.embed = AsyncMock(return_value=_VECTOR)
        ids = await _seed_active(db, 20)

        await embedding_backfill_sweep(db, embeddings, cfg)
        assert embeddings.embed.call_count == 20
        embeddings.embed.reset_mock()

        await embedding_backfill_sweep(db, embeddings, cfg)

        assert embeddings.embed.call_count == 0
        assert await _embedded_ids(db, cfg.embeddings.model) == set(ids)
        await db.close()


@pytest.mark.asyncio
async def test_embed_failure_keeps_cursor_and_later_run_heals_failed_ids():
    with tempfile.TemporaryDirectory() as tmp:
        db = Database(f"{tmp}/runtime.db")
        await db.connect()
        cfg = AppConfig()
        cfg.embeddings.enabled = True
        calls = {"n": 0}

        async def flaky(text: str) -> np.ndarray:
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("embedding upstream down")
            return _VECTOR

        embeddings = AsyncMock()
        embeddings.embed = AsyncMock(side_effect=flaky)
        ids = await _seed_active(db, 5)

        await embedding_backfill_sweep(db, embeddings, cfg)

        # First embed call hits the lowest probe id; the cursor must not have
        # moved past it, and no id may have advanced either.
        assert calls["n"] == 1
        assert await db.get_meta_value(_cursor_key(cfg)) is None
        assert await _embedded_ids(db, cfg.embeddings.model) == set()

        embeddings.embed = AsyncMock(return_value=_VECTOR)
        await embedding_backfill_sweep(db, embeddings, cfg)

        assert await _embedded_ids(db, cfg.embeddings.model) == set(ids)
        assert await db.get_meta_value(_cursor_key(cfg)) == max(ids)
        await db.close()


@pytest.mark.asyncio
async def test_disabled_embeddings_sweep_probes_and_embeds_nothing():
    with tempfile.TemporaryDirectory() as tmp:
        db = Database(f"{tmp}/runtime.db")
        await db.connect()
        cfg = AppConfig()
        cfg.embeddings.enabled = False
        embeddings = AsyncMock()
        embeddings.embed = AsyncMock(return_value=_VECTOR)
        probe_calls = 0
        original_probe = db.vector_backfill_probe

        async def counting_probe(*args, **kwargs):
            nonlocal probe_calls
            probe_calls += 1
            return await original_probe(*args, **kwargs)

        db.vector_backfill_probe = counting_probe

        await embedding_backfill_sweep(db, embeddings, cfg)

        assert probe_calls == 0
        assert embeddings.embed.call_count == 0
        await db.close()

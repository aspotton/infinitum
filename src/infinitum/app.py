from __future__ import annotations

import asyncio
import contextlib
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from . import __version__
from .config import AppConfig, load_config
from .database import Database
from .routes import admin, memory, openai
from .runtime import build_runtime

log = logging.getLogger(__name__)


async def _run_maintenance(db: Database, stop: asyncio.Event) -> None:
    """Purge aged done jobs, then compress legacy raw event rows in bounded batches.

    The stop event is polled between batches instead of cancelling: cancelling
    unwinds the await while the batch's ``to_thread`` may still be writing on the
    shared connection, so ``db.close()`` would then race an orphan thread (a logged
    error on serialized-threading SQLite builds, undefined behaviour on
    non-serialized ones). Stopping cooperatively removes that race by design and
    bounds shutdown to one 8 MB / sub-second batch.
    """
    try:
        await db.purge_done_jobs()
        while not stop.is_set() and await db.compress_events_batch():
            await asyncio.sleep(0.1)
    except Exception:
        log.exception("event maintenance failed")


def create_app(config: AppConfig | None = None) -> FastAPI:
    cfg = config or load_config()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        runtime = await build_runtime(cfg)
        app.state.runtime = runtime
        runtime.worker.start()
        # Closure-local, never awaited here: maintenance must not delay first serving.
        stop_maintenance = asyncio.Event()
        maintenance = asyncio.create_task(
            _run_maintenance(runtime.db, stop_maintenance),
            name="infinitum-event-maintenance",
        )
        try:
            yield
        finally:
            # Stop-and-await, not cancel: cancelling can orphan the batch past db.close().
            stop_maintenance.set()
            with contextlib.suppress(asyncio.CancelledError):
                await maintenance
            await runtime.worker.stop()
            await runtime.upstream.close()
            await runtime.embeddings.close()
            await runtime.db.close()

    app = FastAPI(title="Infinitum", version=__version__, lifespan=lifespan)
    app.include_router(openai.router)
    app.include_router(memory.router)
    app.include_router(admin.router)
    return app


app = create_app()

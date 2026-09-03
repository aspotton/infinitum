from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from .config import AppConfig, load_config
from .routes import admin, memory, openai
from .runtime import build_runtime


def create_app(config: AppConfig | None = None) -> FastAPI:
    cfg = config or load_config()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        runtime = await build_runtime(cfg)
        app.state.runtime = runtime
        runtime.worker.start()
        try:
            yield
        finally:
            await runtime.worker.stop()
            await runtime.upstream.close()
            await runtime.embeddings.close()
            await runtime.db.close()

    app = FastAPI(title="Infinitum", version="0.2.2", lifespan=lifespan)
    app.include_router(openai.router)
    app.include_router(memory.router)
    app.include_router(admin.router)
    return app


app = create_app()

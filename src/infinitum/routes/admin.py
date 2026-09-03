from __future__ import annotations

from fastapi import APIRouter, Query, Request

from ..runtime import Runtime

router = APIRouter()


def _runtime(request: Request) -> Runtime:
    return request.app.state.runtime


@router.get("/health")
async def health(request: Request):
    runtime = _runtime(request)
    return {
        "status": "ok",
        "version": "0.2.1",
        "memory_enabled": runtime.config.memory.enabled,
        "learning_enabled": runtime.config.learning.enabled,
        "embeddings_enabled": runtime.config.embeddings.enabled,
        "fts_enabled": runtime.db.fts_enabled,
    }


@router.get("/events")
async def events(
    request: Request,
    limit: int = Query(100, ge=1, le=1000),
    session_id: str | None = None,
    user_id: str | None = None,
    project_id: str | None = None,
):
    return await _runtime(request).db.list_events(
        limit=limit,
        session_id=session_id,
        user_id=user_id,
        project_id=project_id,
    )


@router.get("/request-context")
async def request_context(request: Request):
    """Inspect how the current request headers resolve without invoking a model."""

    return _runtime(request).request_context.resolve(request.headers)


@router.get("/topics")
async def topics(request: Request, limit: int = Query(100, ge=1, le=1000)):
    return await _runtime(request).db.list_topics(limit=limit)

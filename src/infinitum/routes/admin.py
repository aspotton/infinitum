from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, Request, Response

from .. import __version__
from ..pagination import decode_cursor, encode_cursor
from ..runtime import Runtime


def _keyset(offset: int | None, cursor: str | None) -> tuple[str, str] | None:
    """Resolve offset/cursor params to a keyset (sort_value, tiebreak) pair; 400 on offset or garbage."""
    if offset is not None:
        raise HTTPException(
            400, "offset pagination is not supported; use the cursor from the X-Next-Cursor response header"
        )
    if cursor is None:
        return None
    try:
        return decode_cursor(cursor)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc

router = APIRouter()


def _runtime(request: Request) -> Runtime:
    return request.app.state.runtime


@router.get("/health")
async def health(request: Request):
    runtime = _runtime(request)
    return {
        "status": "ok",
        "version": __version__,
        "memory_enabled": runtime.config.memory.enabled,
        "learning_enabled": runtime.config.learning.enabled,
        "embeddings_enabled": runtime.config.embeddings.enabled,
        "fts_enabled": runtime.db.fts_enabled,
        "active_requests": runtime.active_requests.value,
    }


@router.get("/events")
async def events(
    request: Request,
    response: Response,
    limit: int = Query(100, ge=1, le=1000),
    session_id: str | None = None,
    user_id: str | None = None,
    project_id: str | None = None,
    cursor: str | None = None,
    offset: int | None = None,
):
    before = _keyset(offset, cursor)
    items = await _runtime(request).db.list_events(
        limit=limit + 1,
        session_id=session_id,
        user_id=user_id,
        project_id=project_id,
        before=before,
    )
    if len(items) > limit:
        items = items[:limit]
        response.headers["X-Next-Cursor"] = encode_cursor(items[-1].created_at.isoformat(), items[-1].id)
    return items


@router.get("/request-context")
async def request_context(request: Request):
    """Inspect how the current request headers resolve without invoking a model."""

    return _runtime(request).request_context.resolve(request.headers)


@router.get("/topics")
async def topics(
    request: Request,
    response: Response,
    limit: int = Query(100, ge=1, le=1000),
    cursor: str | None = None,
    offset: int | None = None,
):
    before = _keyset(offset, cursor)
    items = await _runtime(request).db.list_topics(limit=limit + 1, before=before)
    if len(items) > limit:
        items = items[:limit]
        response.headers["X-Next-Cursor"] = encode_cursor(items[-1].updated_at.isoformat(), items[-1].topic)
    return items

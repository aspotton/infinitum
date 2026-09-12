from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, Request, Response

from ..models import Memory, MemoryCreateRequest, MemorySearchRequest
from ..pagination import decode_cursor, encode_cursor
from ..runtime import Runtime

router = APIRouter(prefix="/memory", tags=["memory"])


def _runtime(request: Request) -> Runtime:
    return request.app.state.runtime


@router.get("")
async def list_memories(
    request: Request,
    response: Response,
    limit: int = Query(100, ge=1, le=1000),
    status: str | None = None,
    cursor: str | None = None,
    offset: int | None = None,
):
    if offset is not None:
        raise HTTPException(
            400, "offset pagination is not supported; use the cursor from the X-Next-Cursor response header"
        )
    before = None
    if cursor is not None:
        try:
            before = decode_cursor(cursor)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
    items = await _runtime(request).db.list_memories(limit=limit + 1, status=status, before=before)
    if len(items) > limit:
        items = items[:limit]
        response.headers["X-Next-Cursor"] = encode_cursor(items[-1].updated_at.isoformat(), items[-1].id)
    return items


@router.post("")
async def create_memory(request: Request, body: MemoryCreateRequest):
    runtime = _runtime(request)
    memory = Memory(**body.model_dump())
    await runtime.db.create_memory(memory, evidence_type="manual_admin")
    vector = await runtime.embeddings.embed(memory.content)
    if vector is not None:
        await runtime.db.set_embedding(memory.id, runtime.config.embeddings.model, vector)
    return memory


@router.post("/search")
async def search_memory(request: Request, body: MemorySearchRequest):
    runtime = _runtime(request)
    request_context = runtime.request_context.resolve(request.headers)
    try:
        results = await runtime.retriever.search(
            body.query,
            limit=body.limit,
            request_context=request_context,
            temporal_view=body.temporal_view,
            as_of=body.as_of,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return [item.model_dump() for item in results]


@router.get("/{memory_id}")
async def get_memory(request: Request, memory_id: str):
    runtime = _runtime(request)
    memory = await runtime.db.get_memory(memory_id)
    if not memory:
        raise HTTPException(404, "memory not found")
    observations = await runtime.db.list_observations(memory_id)
    return {**memory.model_dump(), "observations": [obs.model_dump() for obs in observations]}


@router.delete("/{memory_id}")
async def delete_memory(request: Request, memory_id: str):
    if not await _runtime(request).db.archive_memory(memory_id):
        raise HTTPException(404, "memory not found")
    return {"id": memory_id, "status": "archived"}

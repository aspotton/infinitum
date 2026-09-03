from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, Request

from ..models import Memory, MemoryCreateRequest, MemorySearchRequest
from ..runtime import Runtime

router = APIRouter(prefix="/memory", tags=["memory"])


def _runtime(request: Request) -> Runtime:
    return request.app.state.runtime


@router.get("")
async def list_memories(request: Request, limit: int = Query(100, ge=1, le=1000), status: str | None = None):
    return await _runtime(request).db.list_memories(limit=limit, status=status)


@router.post("")
async def create_memory(request: Request, body: MemoryCreateRequest):
    runtime = _runtime(request)
    memory = Memory(**body.model_dump())
    await runtime.db.create_memory(memory)
    vector = await runtime.embeddings.embed(memory.content)
    if vector is not None:
        await runtime.db.set_embedding(memory.id, runtime.config.embeddings.model, vector)
    return memory


@router.post("/search")
async def search_memory(request: Request, body: MemorySearchRequest):
    runtime = _runtime(request)
    request_context = runtime.request_context.resolve(request.headers)
    results = await runtime.retriever.search(
        body.query, limit=body.limit, request_context=request_context
    )
    return [item.model_dump() for item in results]


@router.get("/{memory_id}")
async def get_memory(request: Request, memory_id: str):
    memory = await _runtime(request).db.get_memory(memory_id)
    if not memory:
        raise HTTPException(404, "memory not found")
    return memory


@router.delete("/{memory_id}")
async def delete_memory(request: Request, memory_id: str):
    if not await _runtime(request).db.archive_memory(memory_id):
        raise HTTPException(404, "memory not found")
    return {"id": memory_id, "status": "archived"}

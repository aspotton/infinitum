from __future__ import annotations

import json
from typing import Any

import httpx
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response, StreamingResponse

from ..models import Event, RequestContext, new_id
from ..runtime import Runtime
from ..text import first_text_content
from ..upstream import extract_nonstream_assistant, extract_stream_assistant

router = APIRouter()


def _runtime(request: Request) -> Runtime:
    return request.app.state.runtime


def _session_id(request: Request, body: dict[str, Any]) -> str:
    # OpenCode's OpenAI-compatible provider path commonly supplies X-Session-Id
    # / x-session-affinity; its own provider uses x-opencode-session. Prefer the
    # canonical Infinitum header. The former x-context-* name remains accepted
    # as a compatibility alias for pre-0.2 deployments.
    for name in (
        "x-infinitum-session-id",
        "x-context-session-id",
        "x-opencode-session",
        "x-session-id",
        "x-session-affinity",
    ):
        header = request.headers.get(name)
        if header:
            return header
    metadata = body.get("metadata")
    if isinstance(metadata, dict):
        if metadata.get("infinitum_session_id"):
            return str(metadata["infinitum_session_id"])
        if metadata.get("context_session_id"):
            return str(metadata["context_session_id"])
    return new_id("ses")


def _latest_user(messages: list[dict[str, Any]]) -> str:
    for msg in reversed(messages):
        if msg.get("role") == "user":
            return first_text_content(msg.get("content"))
    return ""


def _response_headers(upstream: httpx.Response, extra: dict[str, str] | None = None) -> dict[str, str]:
    headers: dict[str, str] = {}
    for key in ("x-request-id", "openai-processing-ms", "openai-version"):
        if key in upstream.headers:
            headers[key] = upstream.headers[key]
    if extra:
        headers.update(extra)
    return headers


def _safe_debug_header(value: str) -> str:
    return value.encode("ascii", errors="ignore").decode("ascii")[:240]


async def _record_completion(
    runtime: Runtime,
    *,
    request_id: str,
    session_id: str,
    user_event_id: str,
    user_text: str,
    assistant_text: str,
    assistant_metadata: dict[str, Any],
    model: str,
    learning_enabled: bool,
    request_context: RequestContext,
) -> None:
    assistant_event = Event(
        session_id=session_id,
        user_id=request_context.user_id,
        project_id=request_context.project_id,
        cwd=request_context.cwd,
        request_id=request_id,
        event_type="message.assistant",
        role="assistant",
        content=assistant_text,
        metadata=assistant_metadata,
    )
    await runtime.db.add_event(assistant_event)
    if learning_enabled and runtime.config.learning.enabled:
        await runtime.db.enqueue_job(
            "learn_interaction",
            {
                "request_id": request_id,
                "session_id": session_id,
                "model": model,
                "user_text": user_text,
                "assistant_text": assistant_text,
                "source_event_ids": [user_event_id, assistant_event.id],
                "request_context": request_context.model_dump(),
            },
        )


@router.post("/v1/chat/completions")
async def chat_completions(request: Request) -> Response:
    runtime = _runtime(request)
    try:
        body = await request.json()
    except Exception as exc:
        raise HTTPException(400, f"invalid JSON body: {exc}") from exc
    if not isinstance(body, dict) or not isinstance(body.get("messages"), list):
        raise HTTPException(400, "messages must be an array")

    original_messages = body["messages"]
    model = str(body.get("model") or "")
    session_id = _session_id(request, body)
    request_context = runtime.request_context.resolve(request.headers)
    request_id = new_id("req")
    user_text = _latest_user(original_messages)

    await runtime.db.add_request(
        request_id, session_id, user_text, model, context=request_context
    )
    await runtime.db.add_event(
        Event(
            session_id=session_id,
            user_id=request_context.user_id,
            project_id=request_context.project_id,
            cwd=request_context.cwd,
            request_id=request_id,
            event_type="request.received",
            content=json.dumps(original_messages, ensure_ascii=False),
            metadata={
                "model": model,
                "stream": bool(body.get("stream")),
                "request_context": request_context.compact(),
            },
        )
    )
    user_event = await runtime.db.add_event(
        Event(
            session_id=session_id,
            user_id=request_context.user_id,
            project_id=request_context.project_id,
            cwd=request_context.cwd,
            request_id=request_id,
            event_type="message.user",
            role="user",
            content=user_text,
        )
    )

    def control_header(canonical: str, legacy: str, default: str) -> str:
        return request.headers.get(canonical) or request.headers.get(legacy) or default

    memory_enabled = control_header("x-infinitum-memory", "x-context-memory", "on").lower() not in {
        "off", "false", "0"
    }
    learning_enabled = control_header(
        "x-infinitum-learning", "x-context-learning", "on"
    ).lower() not in {"off", "false", "0"}
    debug = control_header("x-infinitum-debug", "x-context-debug", "false").lower() in {
        "on", "true", "1"
    }

    if memory_enabled:
        compiled = await runtime.compiler.compile(
            original_messages, request_context=request_context
        )
        body["messages"] = runtime.compiler.inject(original_messages, compiled)
    else:
        compiled = None

    if compiled:
        for rank, item in enumerate(compiled.memories, 1):
            token_cost = runtime.compiler.tokens.count_text(runtime.compiler._render_memory(item))
            await runtime.db.add_request_memory(request_id, item.memory.id, rank, item.score, token_cost)

    debug_headers: dict[str, str] = {"x-infinitum-request-id": request_id}
    if debug and compiled:
        debug_headers.update(
            {
                "x-infinitum-memories-injected": str(len(compiled.memories)),
                "x-infinitum-memory-tokens": str(compiled.memory_tokens),
                "x-infinitum-memory-budget": str(compiled.available_memory_tokens),
            }
        )
    if debug:
        if request_context.user_id:
            debug_headers["x-infinitum-resolved-user-id"] = _safe_debug_header(
                request_context.user_id
            )
        if request_context.project_id:
            debug_headers["x-infinitum-resolved-project-id"] = _safe_debug_header(
                request_context.project_id
            )
        debug_headers["x-infinitum-project-derived-from-cwd"] = (
            "true" if request_context.project_derived_from_cwd else "false"
        )

    if body.get("stream"):
        async def completed(raw: bytes) -> None:
            text, meta = extract_stream_assistant(raw)
            stream_complete = b"[DONE]" in raw
            meta["stream_complete"] = stream_complete
            await _record_completion(
                runtime,
                request_id=request_id,
                session_id=session_id,
                user_event_id=user_event.id,
                user_text=user_text,
                assistant_text=text,
                assistant_metadata=meta,
                model=model,
                learning_enabled=learning_enabled and stream_complete,
                request_context=request_context,
            )

        try:
            iterator = await runtime.upstream.stream_bytes(
                "chat/completions",
                body,
                request.headers,
                completed,
                request_context=request_context,
            )
        except httpx.HTTPStatusError as exc:
            response = exc.response
            return Response(
                content=response.content,
                status_code=response.status_code,
                media_type=response.headers.get("content-type", "application/json"),
                headers=debug_headers,
            )
        return StreamingResponse(iterator, media_type="text/event-stream", headers=debug_headers)

    try:
        upstream = await runtime.upstream.request(
            "POST",
            "chat/completions",
            incoming_headers=request.headers,
            json_body=body,
            request_context=request_context,
        )
    except httpx.RequestError as exc:
        raise HTTPException(502, f"upstream connection failed: {exc}") from exc

    if upstream.status_code < 400:
        try:
            parsed = upstream.json()
            assistant_text, assistant_meta = extract_nonstream_assistant(parsed)
        except Exception:
            assistant_text, assistant_meta = "", {}
        await _record_completion(
            runtime,
            request_id=request_id,
            session_id=session_id,
            user_event_id=user_event.id,
            user_text=user_text,
            assistant_text=assistant_text,
            assistant_metadata=assistant_meta,
            model=model,
            learning_enabled=learning_enabled,
            request_context=request_context,
        )

    return Response(
        content=upstream.content,
        status_code=upstream.status_code,
        media_type=upstream.headers.get("content-type", "application/json"),
        headers=_response_headers(upstream, debug_headers),
    )


@router.get("/v1/models")
async def models(request: Request) -> Response:
    runtime = _runtime(request)
    request_context = runtime.request_context.resolve(request.headers)
    try:
        upstream = await runtime.upstream.request(
            "GET",
            "models",
            incoming_headers=request.headers,
            request_context=request_context,
        )
    except httpx.RequestError as exc:
        raise HTTPException(502, f"upstream connection failed: {exc}") from exc
    return Response(
        content=upstream.content,
        status_code=upstream.status_code,
        media_type=upstream.headers.get("content-type", "application/json"),
        headers=_response_headers(upstream),
    )

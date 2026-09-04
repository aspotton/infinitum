from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response, StreamingResponse

from .. import memory_tools
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

    ours_injected: set[str] = set()
    if memory_enabled:
        compiled = await runtime.compiler.compile(
            original_messages, request_context=request_context
        )
        # Inject our tool defs + hint BEFORE inject(): inject copies the message
        # list and embeds compiled.text at call time, so mutating either after
        # this point would be dead.
        if (
            memory_enabled
            and runtime.config.memory.enabled
            and runtime.config.memory.tools_enabled
            and compiled.text
        ):
            ours_injected = set(memory_tools.injected_tool_names(body.get("tools")))
            if ours_injected:
                body["tools"] = (
                    list(body.get("tools") or [])
                    + memory_tools.build_tool_defs(ours_injected)
                )
                if compiled.memories:
                    compiled.text += (
                        "\n\nDeeper detail is available via the "
                        + " and ".join(sorted(ours_injected))
                        + " tools using the memory ids above."
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

        # stream_bytes binds on_complete at call time but the classifier only
        # decides while consuming, so every iteration passes None and the
        # terminal iteration's recording happens in the returned generator's
        # finally: suppressed iterations never reach a generator, so they never
        # record. The loop runs inside the handler; only the terminal
        # iteration's bytes ever reach the client.
        tool_rounds = 0
        final_iterator: AsyncIterator[bytes] | None = None
        for _ in range(memory_tools.MAX_ITERATIONS):
            try:
                iterator = await runtime.upstream.stream_bytes(
                    "chat/completions",
                    body,
                    request.headers,
                    None,
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

            classifier = memory_tools.StreamClassifier(ours_injected)
            accum = bytearray()
            exhausted = True
            try:
                async for chunk in iterator:
                    accum.extend(chunk)
                    classifier.feed(chunk)
                    if classifier.decide() == "passthrough":
                        exhausted = False
                        break
            except httpx.RequestError as exc:
                raise HTTPException(502, f"upstream connection failed: {exc}") from exc

            decision = "passthrough" if not exhausted else classifier.finish()
            if decision in ("passthrough", "replay"):
                live: AsyncIterator[bytes] | None = iterator
                if decision == "replay":
                    # Drain the rest here (its finally closes the upstream
                    # response); the full bytes replay from accum below.
                    try:
                        async for chunk in iterator:
                            accum.extend(chunk)
                    except httpx.RequestError as exc:
                        raise HTTPException(
                            502, f"upstream connection failed: {exc}"
                        ) from exc
                    live = None

                async def stream_out(
                    accum=accum, live=live
                ) -> AsyncIterator[bytes]:  # default-args pin this iteration's state
                    try:
                        if accum:
                            yield bytes(accum)
                        if live is not None:
                            async for chunk in live:
                                accum.extend(chunk)
                                yield chunk
                    finally:
                        await completed(bytes(accum))

                final_iterator = stream_out()
                break

            tool_rounds += 1
            calls = memory_tools.reassemble_stream_tool_calls(classifier.calls)
            assistant_content, _ = extract_stream_assistant(classifier.collected_bytes())
            assistant_message: dict[str, Any] = {"role": "assistant"}
            if assistant_content:
                assistant_message["content"] = assistant_content
            assistant_message["tool_calls"] = calls
            body["messages"] = body["messages"] + [assistant_message]
            for call in calls:
                arguments = call["function"].get("arguments", "")
                result = await memory_tools.execute(
                    call["function"]["name"], arguments, runtime, request_context
                )
                await runtime.db.add_event(
                    Event(
                        session_id=session_id,
                        user_id=request_context.user_id,
                        project_id=request_context.project_id,
                        cwd=request_context.cwd,
                        request_id=request_id,
                        event_type="memory.tool_call",
                        role="tool",
                        content=arguments,
                        metadata={
                            "name": call["function"]["name"],
                            "result_chars": len(result),
                            "tool_call_id": call.get("id"),
                            "assistant_message": assistant_message,
                        },
                    )
                )
                body["messages"] = body["messages"] + [
                    {"role": "tool", "tool_call_id": call.get("id"), "content": result}
                ]

        if debug and (ours_injected or tool_rounds):
            debug_headers["x-infinitum-memory-tool-calls"] = str(tool_rounds)
        if final_iterator is None:
            # Cap exhausted with every round suppressed: dead branch (clients
            # never invoke our tools directly); emit an empty SSE stream.
            final_iterator = iter(())
        return StreamingResponse(
            final_iterator, media_type="text/event-stream", headers=debug_headers
        )

    tool_rounds = 0
    for _ in range(memory_tools.MAX_ITERATIONS):
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

        if upstream.status_code >= 400:
            break  # terminal: forward verbatim, no recording (mirrors the old guard)

        try:
            parsed = upstream.json()
        except Exception:
            parsed = None  # terminal; records ("", {}) like the old fallback

        calls = [
            tool_call
            for choice in ((parsed or {}).get("choices") or [])[:1]
            for tool_call in (choice.get("message") or {}).get("tool_calls") or []
        ]
        if parsed is None or not memory_tools.classify_tool_calls(calls, ours_injected):
            try:
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
            break

        tool_rounds += 1
        assistant_message = (parsed.get("choices") or [{}])[0].get("message") or {
            "role": "assistant",
            "tool_calls": calls,
        }
        body["messages"] = body["messages"] + [assistant_message]
        for call in calls:
            arguments = call["function"].get("arguments", "")
            result = await memory_tools.execute(
                call["function"]["name"], arguments, runtime, request_context
            )
            await runtime.db.add_event(
                Event(
                    session_id=session_id,
                    user_id=request_context.user_id,
                    project_id=request_context.project_id,
                    cwd=request_context.cwd,
                    request_id=request_id,
                    event_type="memory.tool_call",
                    role="tool",
                    content=arguments,
                    metadata={
                        "name": call["function"]["name"],
                        "result_chars": len(result),
                        "tool_call_id": call.get("id"),
                        "assistant_message": assistant_message,
                    },
                )
            )
            body["messages"] = body["messages"] + [
                {"role": "tool", "tool_call_id": call.get("id"), "content": result}
            ]
    else:
        pass  # cap exhausted: forward the last tool-call response verbatim, unrecorded

    if debug and (ours_injected or tool_rounds):
        debug_headers["x-infinitum-memory-tool-calls"] = str(tool_rounds)

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

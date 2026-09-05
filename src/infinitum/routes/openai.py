from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator
from dataclasses import replace
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


def _session_id(request: Request, body: dict[str, Any]) -> str | None:
    """Return the client-supplied session id, or None when the client sent none.

    A None return means "no client session"; callers that need an id must
    generate a provenance-only id themselves rather than treating a generated
    id as if the client had pinned a session.
    """
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
    return None


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


def _strip_injected_tools(body: dict[str, Any], names: set[str]) -> None:
    """Remove the tool defs we injected from the forwarded body, in place.

    Used by the forced terminal round past the tool-loop cap, alongside
    tool_choice:"none": our defs gone means a server that keeps calling tools
    can only reach client tools or nothing at all. Client tools and malformed
    entries are preserved untouched.
    """
    tools = body.get("tools")
    if not isinstance(tools, list):
        return
    kept: list[Any] = []
    for tool in tools:
        name = tool.get("function", {}).get("name") if isinstance(tool, dict) else None
        if isinstance(name, str) and name in names:
            continue
        kept.append(tool)
    if kept:
        body["tools"] = kept
    else:
        body.pop("tools", None)


def _synthesize_from_tool_results(body: dict[str, Any], model: str) -> dict[str, Any]:
    """Build a plain assistant chat-completion from the gathered tool results.

    Last-resort fallback when the forced terminal round still returns a blank
    or tool-call-only response (server ignored tool_choice:"none"). Concatenates
    the JSON tool results already appended to the transcript (role:"tool") into a
    single assistant message so the client receives content, never a blank or a
    dangling tool_call. No extra upstream call.
    """
    parts = [
        str(message.get("content") or "")
        for message in body.get("messages", [])
        if message.get("role") == "tool"
    ]
    content = "Based on the retrieved memories:\n\n" + "\n\n".join(parts)
    return {
        "id": new_id("chatcmpl"),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
    }


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

    # Hallucination-guard state, computed once per request BEFORE any def
    # injection so client_names reflects the original inbound tool list.
    client_names = memory_tools.client_tool_names(body.get("tools"))
    guard_active = (
        memory_enabled
        and runtime.config.memory.enabled
        and runtime.config.memory.tools_enabled
    )
    reject_count = 0

    ours_injected: set[str] = set()
    if memory_enabled:
        compiled = await runtime.compiler.compile(
            original_messages, request_context=request_context, session_id=session_id
        )
        # Inject our tool defs + hint BEFORE inject(): inject copies the message
        # list and embeds compiled.text at call time, so mutating either after
        # this point would be dead. Tool defs are exposed statically (flag-only
        # gate) so the tools region never flaps between turns and stays cacheable.
        if runtime.config.memory.enabled and runtime.config.memory.tools_enabled:
            ours_injected = set(memory_tools.injected_tool_names(body.get("tools")))
            if ours_injected:
                body["tools"] = (
                    list(body.get("tools") or [])
                    + memory_tools.build_tool_defs(ours_injected)
                )
                if compiled.memories:
                    # Copy-on-write: compile() may hand back a cached block, so
                    # never append the hint into the object the cache still holds.
                    compiled = replace(
                        compiled,
                        text=compiled.text
                        + "\n\nDeeper detail is available via the "
                        + " and ".join(sorted(ours_injected))
                        + " tools using the memory ids above."
                        + " Those are the only memory tools that exist; use them"
                        + " for any memory lookup; never invent another memory"
                        + " tool name.",
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
        ours_active: set[str] = set(ours_injected)
        # One extra round past the tool-round cap: suppress rounds are now
        # reachable on any request (static tool exposure), so the cap can
        # actually fire; the forced round must answer, never blank the client.
        for round_no in range(memory_tools.MAX_ITERATIONS + 1):
            is_forced = round_no == memory_tools.MAX_ITERATIONS
            if is_forced and ours_injected:
                # Past the cap: force a terminal ANSWER round. Stripping defs
                # alone is insufficient for servers with an automatic tool-call
                # parser (vLLM/Qwen) that re-emit our tool names unprompted.
                # tool_choice:"none" (OpenAI spec, honored by the verified vLLM
                # upstream) forbids tool calls so the round must produce text.
                # Keep our names in ours_active so a stray call that a server
                # emits anyway is OUR loop-capped call to suppress, not a
                # foreign tool to forward to the client.
                _strip_injected_tools(body, ours_injected)
                body["tool_choice"] = "none"
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

            classifier = memory_tools.StreamClassifier(
                ours_active, client_names=client_names, guard=guard_active
            )
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
                function = call.get("function") or {}
                name = function.get("name")
                arguments = function.get("arguments", "")
                extra: dict[str, Any] = {}
                if name in ours_active:
                    result = await memory_tools.execute(
                        name, arguments, runtime, request_context
                    )
                else:
                    # Hallucinated memory-tool name: NEVER forwarded to the
                    # client. The model is told the real tool set and the loop
                    # continues (same rule as the non-stream path).
                    result = memory_tools.build_reject_result(name, sorted(ours_injected))
                    reject_count += 1
                    extra = {"rejected": True, "result": result}
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
                            "name": name,
                            "result_chars": len(result),
                            "tool_call_id": call.get("id"),
                            "assistant_message": assistant_message,
                            **extra,
                        },
                    )
                )
                body["messages"] = body["messages"] + [
                    {"role": "tool", "tool_call_id": call.get("id"), "content": result}
                ]

        if debug and (ours_injected or tool_rounds):
            debug_headers["x-infinitum-memory-tool-calls"] = str(tool_rounds)
        if debug and reject_count:
            debug_headers["x-infinitum-memory-tool-rejects"] = str(reject_count)
        if final_iterator is None:
            # Residual: only reachable if the server ignored
            # tool_choice:"none" AND the forced round emitted a suppressible
            # all-ours tool-call stream with no content. Keep the empty-stream
            # fallback; re-streaming synthesis is deliberately not attempted.
            final_iterator = iter(())
        return StreamingResponse(
            final_iterator, media_type="text/event-stream", headers=debug_headers
        )

    tool_rounds = 0
    synthesized = False
    ours_active: set[str] = set(ours_injected)
    for round_no in range(memory_tools.MAX_ITERATIONS + 1):
        is_forced = round_no == memory_tools.MAX_ITERATIONS
        if is_forced and ours_injected:
            # Past the cap: force a terminal ANSWER round. Stripping defs alone
            # is insufficient for servers with an automatic tool-call parser
            # (vLLM/Qwen) that re-emit our tool names unprompted.
            # tool_choice:"none" (OpenAI spec, honored by the verified vLLM
            # upstream) forbids tool calls so the round must produce text. Keep
            # our names in ours_active so a stray call that a server emits
            # anyway is OUR loop-capped call to suppress, not a foreign tool to
            # forward to the client.
            _strip_injected_tools(body, ours_injected)
            body["tool_choice"] = "none"
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

        if is_forced and parsed is not None:
            # Defensive: a server that ignored tool_choice:"none" can still
            # blank the round. Synthesize so the terminal branch below records
            # and forwards a non-empty answer, never null content or a dangling
            # tool_call (the calls list below is recomputed from the
            # synthesized message, which ends the loop).
            try:
                forced_blank = not extract_nonstream_assistant(parsed)[0].strip()
            except Exception:
                forced_blank = True
            if forced_blank:
                parsed = _synthesize_from_tool_results(body, model)
                synthesized = True

        calls = [
            tool_call
            for choice in ((parsed or {}).get("choices") or [])[:1]
            for tool_call in (choice.get("message") or {}).get("tool_calls") or []
        ]
        classified = parsed is not None and memory_tools.classify_tool_calls(calls, ours_active)
        # Partition rule: a non-forced round is a REJECT round only when every
        # call is one of ours or a hallucinated infinitum_* name. Any
        # client-defined/foreign name keeps the terminal-forward contract
        # verbatim (never swallow a client's tool contract).
        reject_round = (
            not classified
            and guard_active
            and not is_forced
            and bool(calls)
            and all(
                (name := (call.get("function") or {}).get("name")) in ours_active
                or memory_tools.is_rejectable_memory_name(name, ours_active, client_names)
                for call in calls
            )
        )
        if not (classified or reject_round):
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
            function = call.get("function") or {}
            name = function.get("name")
            arguments = function.get("arguments", "")
            extra: dict[str, Any] = {}
            if name in ours_active:
                result = await memory_tools.execute(name, arguments, runtime, request_context)
            else:
                # Hallucinated memory-tool name: NEVER forwarded to the client.
                # The model is told the real tool set and the loop continues.
                result = memory_tools.build_reject_result(name, sorted(ours_injected))
                reject_count += 1
                extra = {"rejected": True, "result": result}
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
                        "name": name,
                        "result_chars": len(result),
                        "tool_call_id": call.get("id"),
                        "assistant_message": assistant_message,
                        **extra,
                    },
                )
            )
            body["messages"] = body["messages"] + [
                {"role": "tool", "tool_call_id": call.get("id"), "content": result}
            ]
    if debug and (ours_injected or tool_rounds):
        debug_headers["x-infinitum-memory-tool-calls"] = str(tool_rounds)
    if debug and reject_count:
        debug_headers["x-infinitum-memory-tool-rejects"] = str(reject_count)

    if synthesized:
        return Response(
            content=json.dumps(parsed).encode(),
            status_code=200,
            media_type="application/json",
            headers=debug_headers,
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

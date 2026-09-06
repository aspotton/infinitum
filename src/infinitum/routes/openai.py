from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import replace
from typing import Any

import httpx
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response, StreamingResponse

from .. import memory_tools
from ..compiler import strip_memory_block
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


async def _counted(stream: AsyncIterator[bytes], counter: Any) -> AsyncIterator[bytes]:
    # Decrement via generator finally: covers normal [DONE] completion, client
    # disconnect (GeneratorExit from the ASGI server's aclose()), and mid-stream
    # errors. Runs after the inner on_complete -> enqueue, so the follow-up
    # learning job is queued while the counter is still held.
    try:
        async for chunk in stream:
            yield chunk
    finally:
        counter.decrement()


async def _prefixed(first: bytes | None, rest: AsyncIterator[bytes]) -> AsyncIterator[bytes]:
    # Re-emit a prefetched first generator fragment, then drain the rest.
    # The route prefetches so generator-raised errors (_VerbatimResponse,
    # httpx.RequestError) can still become plain Responses before the
    # StreamingResponse exists; the fragment must never be dropped or doubled.
    if first is not None:
        yield first
    async for item in rest:
        yield item


async def _terminal_stream(
    accum: bytearray,
    live: AsyncIterator[bytes] | None,
    comments: bytes,
    completed: Callable[[bytes], Awaitable[None]],
) -> AsyncIterator[bytes]:
    """Emit a decided round: replay/passthrough bytes, then trailing debug comments.

    `accum` holds the bytes already fed before the decision; `live` is the raw
    upstream iterator, present only for a mid-round passthrough (None for a
    replay whose rest was drained into `accum` in the route body). `completed`
    is awaited in the finally so normal completion AND client disconnect each
    record exactly once, exactly like the pre-0.3 stream_out.
    """
    try:
        if accum:
            yield bytes(accum)
        if live is not None:
            async for chunk in live:
                accum.extend(chunk)
                yield chunk
        if comments:
            yield comments
    finally:
        await completed(bytes(accum))


class _VerbatimResponse(Exception):
    """Carries a rounds-2+ upstream HTTPStatusError to the route for re-presentation.

    Raised inside the Phase-B generator ONLY when zero client bytes have been
    forwarded: the route prefetches the first fragment and converts this into
    the same plain Response a round-1 4xx produces (status/content/media_type
    from the error response, debug headers, no upstream headers passthrough).
    """

    def __init__(self, response: httpx.Response) -> None:
        self.response = response
        super().__init__(str(response.status_code))


# The ONLY allowed byte synthesis on the streaming path: once client bytes exist
# the HTTP status cannot change, so an upstream failure surfaces in-stream.
_UPSTREAM_ERROR_EVENT = (
    b'data: {"error": {"message": "upstream failed", "type": "upstream_error"}}\n\n'
    b"data: [DONE]\n\n"
)


def _debug_stream_comments(
    debug: bool, ours_injected: set[str], tool_rounds: int, reject_count: int
) -> bytes:
    """Trailing SSE comments carrying the debug counters on streaming responses.

    Rounds 2+ run inside the returned StreamingResponse, where headers are
    physically impossible, so both modes carry the same information the
    non-stream path puts in headers, under the same gate. They ride AFTER
    [DONE], so SDKs that stop at the sentinel never see them; the zero-count
    case with no injected tools stays silent exactly like the header did.
    """
    if not debug or not (ours_injected or tool_rounds):
        return b""
    comments = f": x-infinitum-memory-tool-calls {tool_rounds}\n"
    if reject_count:
        comments += f": x-infinitum-memory-tool-rejects {reject_count}\n"
    return comments.encode()


def _stream_assistant_message(classifier: memory_tools.StreamClassifier) -> dict[str, Any]:
    """Build the transcript assistant message from one suppressed stream round.

    Shared by the Phase-A body assembly and the Phase-B generator: merges the
    SSE tool-call deltas and attaches whatever content the round also streamed.
    Reasoning is never included: extract_stream_assistant reads content and
    tool_calls only.
    """
    calls = memory_tools.reassemble_stream_tool_calls(classifier.calls)
    assistant_content, _ = extract_stream_assistant(classifier.collected_bytes())
    assistant_message: dict[str, Any] = {"role": "assistant"}
    if assistant_content:
        assistant_message["content"] = assistant_content
    assistant_message["tool_calls"] = calls
    return assistant_message


async def _run_tool_round(
    runtime: Runtime,
    body: dict[str, Any],
    assistant_message: dict[str, Any],
    *,
    ours_active: set[str],
    ours_injected: set[str],
    request_id: str,
    session_id: str,
    request_context: RequestContext,
) -> int:
    """Execute one suppressed streaming round and append results to the body.

    Extracted verbatim from the pre-0.3 streaming loop's assembly step so the
    Phase-A body and the Phase-B generator share one implementation: execute
    our calls, reject-instruct hallucinated memory names (NEVER forwarded to
    the client), record one memory.tool_call event per call with the
    copy-on-write sanitized assistant message, and append each tool result to
    the transcript. Returns the number of rejected calls this round.
    """
    body["messages"] = body["messages"] + [assistant_message]
    rejected = 0
    for call in assistant_message["tool_calls"]:
        function = call.get("function") or {}
        name = function.get("name")
        arguments = function.get("arguments", "")
        extra: dict[str, Any] = {}
        if name in ours_active:
            result = await memory_tools.execute(name, arguments, runtime, request_context)
        else:
            # Hallucinated memory-tool name: NEVER forwarded to the client.
            # The model is told the real tool set and the loop continues
            # (same rule as the non-stream path).
            result = memory_tools.build_reject_result(name, sorted(ours_injected))
            rejected += 1
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
                    # Copy-on-write: sanitize only the durable metadata
                    # copy; the forwarded body["messages"] stays raw.
                    "assistant_message": {
                        **assistant_message,
                        "content": strip_memory_block(
                            assistant_message.get("content") or ""
                        ),
                    },
                    **extra,
                },
            )
        )
        body["messages"] = body["messages"] + [
            {"role": "tool", "tool_call_id": call.get("id"), "content": result}
        ]
    return rejected


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
    assistant_text = strip_memory_block(assistant_text)
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
    counter = runtime.active_requests
    try:
        body = await request.json()
    except Exception as exc:
        raise HTTPException(400, f"invalid JSON body: {exc}") from exc
    if not isinstance(body, dict) or not isinstance(body.get("messages"), list):
        raise HTTPException(400, "messages must be an array")

    original_messages = body["messages"]
    model = str(body.get("model") or "")
    client_session_id = _session_id(request, body)  # str | None - client-supplied only
    session_id = client_session_id or new_id("ses")  # provenance id for events/audit
    request_context = runtime.request_context.resolve(request.headers)
    request_id = new_id("req")
    user_text = strip_memory_block(_latest_user(original_messages))

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
            original_messages, request_context=request_context, session_id=client_session_id
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
                    # ponytail: this footer literal is coupled across three sites
                    # (compiler._block_body wrapper, this route hint, compiler._FOOTER_RE);
                    # change one, change all.
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

        # Two-phase streaming. Phase A decides round 1 in the route body,
        # observable-identical to the pre-0.3 loop (success AND failure), so
        # round-1 4xx/transport errors keep returning plain Responses and
        # test_qa_k passes untouched. Phase B is ONE shared generator that runs
        # upstream attempts 2..MAX+1 inside the returned StreamingResponse;
        # `stream_mode` only flips that generator's forwarding policy (live
        # tees visible pre-decision lines, buffered holds everything until the
        # round decision). Phase A never forwards pre-decision bytes in either
        # mode. Debug counters ride the stream as trailing SSE comments because
        # rounds 2+ start before any headers exist.
        stream_mode = runtime.config.memory.stream_reasoning
        fields = runtime.config.memory.reasoning_delta_fields

        async def _rounds_stream(tool_rounds: int, reject_count: int) -> AsyncIterator[bytes]:
            # attempts 2..MAX_ITERATIONS+1: today's total upstream calls per
            # request, minus the round-1 decision already taken in the body.
            recorded = False  # completed() exactly-once guard
            in_terminal = False  # current round resolved terminal (record on disconnect)
            sent_bytes = False  # any byte already written to this client response
            accum = bytearray()
            try:
                for attempt in range(2, memory_tools.MAX_ITERATIONS + 2):
                    is_forced = attempt == memory_tools.MAX_ITERATIONS + 1
                    if is_forced and ours_injected:
                        # Past the cap: force a terminal ANSWER round. Stripping
                        # defs alone is insufficient for servers with an
                        # automatic tool-call parser that re-emits our tool names
                        # unprompted. tool_choice:"none" (OpenAI spec, honored by
                        # many OpenAI-compatible servers) forbids tool calls so the
                        # round must produce text. Keep our names in ours_active so
                        # a stray call a server emits anyway is OUR loop-capped
                        # call to suppress, not a foreign tool to forward.
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
                        # An open failure precedes any bytes of this round, and
                        # before the first forwarded fragment the route can
                        # still re-present it verbatim.
                        recorded = True
                        if sent_bytes:
                            yield _UPSTREAM_ERROR_EVENT
                            return
                        raise _VerbatimResponse(exc.response) from exc
                    classifier = memory_tools.StreamClassifier(
                        ours_active,
                        client_names=client_names,
                        guard=guard_active,
                        reasoning_fields=fields,
                        tee_forward_enabled=(stream_mode == "live"),
                    )
                    accum = bytearray()
                    try:
                        async for chunk in iterator:
                            accum.extend(chunk)
                            out = classifier.consume(chunk)
                            if out:
                                sent_bytes = True
                                yield out
                            if classifier.decide() == "passthrough":
                                in_terminal = True
                    except httpx.RequestError:
                        # Zero forwarded bytes ⇒ the route body (or its prefetch)
                        # still owns the response and maps this to today's 502;
                        # otherwise the status ship has sailed: surface in-stream,
                        # unrecorded.
                        recorded = True
                        if sent_bytes:
                            yield _UPSTREAM_ERROR_EVENT
                            return
                        raise
                    decision = classifier.decide()
                    if decision == "undecided":
                        decision = classifier.finish()
                    if decision == "suppress":
                        # A tool round leaks nothing: bytes held past the freeze
                        # point are discarded and nothing is recorded; assemble
                        # the round's tool results and run the next attempt.
                        tool_rounds += 1
                        assistant_message = _stream_assistant_message(classifier)
                        reject_count += await _run_tool_round(
                            runtime,
                            body,
                            assistant_message,
                            ours_active=ours_active,
                            ours_injected=ours_injected,
                            request_id=request_id,
                            session_id=session_id,
                            request_context=request_context,
                        )
                        continue
                    in_terminal = True
                    if classifier.forwarded:
                        # Live tee: visible lines already streamed; release the
                        # frozen tail (or nothing, if passthrough decided early).
                        held = classifier.flush_held()
                        if held:
                            yield held
                    elif accum:
                        # Buffered hold-all (or a live round with no visible
                        # lines): replay this round's bytes verbatim, once.
                        yield bytes(accum)
                    comments = _debug_stream_comments(
                        debug, ours_injected, tool_rounds, reject_count
                    )
                    if comments:
                        yield comments
                    recorded = True
                    await completed(bytes(accum))
                    return
                # Cap exhausted: every round suppressed. Mirrors today's
                # residual empty terminal; nothing is ever recorded here.
                comments = _debug_stream_comments(
                    debug, ours_injected, tool_rounds, reject_count
                )
                if comments:
                    yield comments
            finally:
                # Only client-disconnect teardown of a TERMINAL round records a
                # partial here (parity with the old stream_out finally): error
                # and suppressed rounds set `recorded` or never set
                # `in_terminal`, so they never produce assistant events.
                if not recorded and in_terminal:
                    await completed(bytes(accum))

        counter.increment()
        handed_off = False
        try:
            tool_rounds = 0
            ours_active: set[str] = set(ours_injected)
            # Phase A: round-1 decision in the body. tee_forward_enabled=False
            # keeps live byte-identical to buffered here: hold everything, decide
            # on content/foreign or at end of stream.
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
                ours_active,
                client_names=client_names,
                guard=guard_active,
                reasoning_fields=fields,
                tee_forward_enabled=False,
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
                    # response); the full bytes replay from accum in the
                    # generator below.
                    try:
                        async for chunk in iterator:
                            accum.extend(chunk)
                    except httpx.RequestError as exc:
                        raise HTTPException(
                            502, f"upstream connection failed: {exc}"
                        ) from exc
                    live = None
                handed_off = True
                return StreamingResponse(
                    _counted(
                        _terminal_stream(
                            accum,
                            live,
                            _debug_stream_comments(
                                debug, ours_injected, tool_rounds, reject_count
                            ),
                            completed,
                        ),
                        counter,
                    ),
                    media_type="text/event-stream",
                    headers=debug_headers,
                )

            # Round 1 suppressed: assemble it in the body, then hand every
            # remaining attempt to the Phase-B generator.
            tool_rounds += 1
            assistant_message = _stream_assistant_message(classifier)
            reject_count += await _run_tool_round(
                runtime,
                body,
                assistant_message,
                ours_active=ours_active,
                ours_injected=ours_injected,
                request_id=request_id,
                session_id=session_id,
                request_context=request_context,
            )

            # Prefetch one fragment so generator-raised plumbing errors
            # (_VerbatimResponse, raw RequestError) still become today's plain
            # Response / 502 BEFORE the StreamingResponse exists.
            stream = _rounds_stream(tool_rounds, reject_count)
            try:
                first: bytes | None = await stream.__anext__()
            except StopAsyncIteration:
                first = None
            except _VerbatimResponse as verbatim:
                await stream.aclose()
                failed = verbatim.response
                return Response(
                    content=failed.content,
                    status_code=failed.status_code,
                    media_type=failed.headers.get("content-type", "application/json"),
                    headers=debug_headers,
                )
            except httpx.RequestError as exc:
                await stream.aclose()
                raise HTTPException(502, f"upstream connection failed: {exc}") from exc
            handed_off = True
            return StreamingResponse(
                _counted(_prefixed(first, stream), counter),
                media_type="text/event-stream",
                headers=debug_headers,
            )
        finally:
            if not handed_off:
                counter.decrement()

    counter.increment()
    try:
        tool_rounds = 0
        synthesized = False
        ours_active: set[str] = set(ours_injected)
        for round_no in range(memory_tools.MAX_ITERATIONS + 1):
            is_forced = round_no == memory_tools.MAX_ITERATIONS
            if is_forced and ours_injected:
                # Past the cap: force a terminal ANSWER round. Stripping defs alone
                # is insufficient for servers with an automatic tool-call parser
                # parser that re-emits our tool names unprompted.
                # tool_choice:"none" (OpenAI spec, honored by many OpenAI-compatible
                # servers) forbids tool calls so the round must produce text. Keep
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
                            "assistant_message": {
                                **assistant_message,
                                "content": strip_memory_block(
                                    assistant_message.get("content") or ""
                                ),
                            },
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
    finally:
        counter.decrement()


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

"""Read-only memory drill-down tools exposed to the model when tools_enabled.

These executors never mutate state: `retriever.search` and `db.get_memory` are
read-only. Model-caused errors return error-JSON strings instead of raising so
the server-side tool loop can feed them back to the model.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from typing import Any, Protocol

TOOL_NAMES = ("infinitum_memory_search", "infinitum_memory_get")
MAX_RESULT_CHARS = 8000
MAX_SEARCH_LIMIT = 50
MAX_ITERATIONS = 4  # tool-loop cap; referenced by routes, hardcoded by design

TOOL_DEFS: dict[str, dict[str, Any]] = {
    "infinitum_memory_search": {
        "type": "function",
        "function": {
            "name": "infinitum_memory_search",
            "description": (
                "Read-only drill-down search over persistent memory. "
                "Returns ranked memories with ids, types, topics, and content."
                " This is the complete set of memory tools: infinitum_memory_search"
                " and infinitum_memory_get; never call another memory tool name."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What to look up in memory."},
                    "limit": {
                        "type": "integer",
                        "description": (
                            f"Maximum results to return (default 10, max {MAX_SEARCH_LIMIT})."
                        ),
                    },
                },
                "required": ["query"],
            },
        },
    },
    "infinitum_memory_get": {
        "type": "function",
        "function": {
            "name": "infinitum_memory_get",
            "description": (
                "Read-only drill-down fetch of one persistent memory by id, "
                "including provenance event ids."
                " This is the complete set of memory tools: infinitum_memory_search"
                " and infinitum_memory_get; never call another memory tool name."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "memory_id": {"type": "string", "description": "Id of the memory to fetch."},
                },
                "required": ["memory_id"],
            },
        },
    },
}


class StreamClassifier:
    """Byte-boundary-safe SSE classifier for the tool-loop holdback.

    Network chunks may split a `data:` line at any byte, so only complete
    (newline-terminated) data lines are JSON-parsed; the partial tail stays in
    the buffer. Content or a foreign function name decides "passthrough" the
    moment it is seen; at end of stream a content-free all-ours tool-call stream
    is "suppress" (run the loop), anything else is "replay" (terminal). Every
    fed byte is accumulated for verbatim replay. With `guard`, hallucinated
    infinitum_* names (not ours, not client-defined) are neither foreign nor
    leaked: they decide "suppress" so the caller can reject-instruct them.
    """

    def __init__(
        self,
        ours: Iterable[str],
        client_names: set[str] | None = None,
        guard: bool = False,
        *,
        reasoning_fields: Sequence[str] = (),
        tee_forward_enabled: bool = False,
    ) -> None:
        self._ours = set(ours)
        self._client_names = client_names or set()
        self._guard = guard
        self._line_buf = bytearray()
        self._held: list[bytes] = []
        self._collected = bytearray()
        self._passthrough = False
        self._content_seen = False
        self._foreign_seen = False
        self._chunks: list[dict[str, Any]] = []
        self._finish_reason: str | None = None
        self._reasoning_fields = tuple(reasoning_fields)
        self._tee_forward = tee_forward_enabled
        self._reasoning_seen = False
        self._frozen = False
        self._fwd_pos = 0
        self._tee_buf = bytearray()
        self._pending: list[bytes] = []

    def feed(self, chunk: bytes) -> list[bytes]:
        """Consume one network chunk; return raw bytes safe to forward now.

        Returns nothing while the decision is undecided; on the feed that turns
        the decision into "passthrough", returns every held chunk so far.
        """
        self._collected.extend(chunk)
        if self._passthrough:
            return [chunk]
        self._held.append(chunk)
        self._line_buf.extend(chunk)
        while (newline := self._line_buf.find(b"\n")) >= 0:
            line = bytes(self._line_buf[:newline])
            del self._line_buf[: newline + 1]
            self._scan_line(line)
        if self._content_seen or self._foreign_seen:
            self._passthrough = True
            flushed, self._held = list(self._held), []
            return flushed
        return []

    def consume(self, chunk: bytes) -> bytes:
        """Tee-mode consume for the Phase-B tool rounds; return bytes to forward now.

        Additive to `feed()`: while undecided it holds all bytes exactly like
        `feed()` unless `tee_forward_enabled` and a reasoning delta has been
        seen, in which case complete visible lines (no tool_calls, null
        finish_reason, not [DONE]) forward verbatim until the first freeze
        trigger line (tool_calls, non-null finish_reason, [DONE], or an
        unparseable data line), whose bytes and all later bytes stay held until
        the round decision. On a "passthrough" decision it flushes every held
        byte in order and forwards raw thereafter, byte-identical to `feed()`.
        """
        self._collected.extend(chunk)
        if self._passthrough:
            self._fwd_pos = len(self._collected)
            return chunk
        self._tee_buf.extend(chunk)
        while (newline := self._tee_buf.find(b"\n")) >= 0:
            line = bytes(self._tee_buf[: newline + 1])
            del self._tee_buf[: newline + 1]
            self._scan_line(line)
            self._pending.append(line)
        if self._content_seen or self._foreign_seen:
            self._passthrough = True
            return self.flush_held()
        if self._tee_forward and self._reasoning_seen and not self._frozen:
            out = bytearray()
            while self._pending and not _freeze_trigger(self._pending[0]):
                line = self._pending.pop(0)
                out.extend(line)
                self._fwd_pos += len(line)
            if self._pending:
                self._frozen = True
            return bytes(out)
        return b""

    def flush_held(self) -> bytes:
        """Return (and mark forwarded) every collected byte not yet forwarded."""
        flushed = bytes(self._collected[self._fwd_pos :])
        self._fwd_pos = len(self._collected)
        self._pending.clear()
        return flushed

    def decide(self) -> str:
        """Mid-stream decision after a feed: "passthrough" or "undecided"."""
        return "passthrough" if self._passthrough else "undecided"

    def finish(self) -> str:
        """Resolve the decision at end of stream: suppress (loop) or replay."""
        if self._passthrough or self._content_seen or self._foreign_seen:
            return "passthrough"
        calls = reassemble_stream_tool_calls(self._chunks)
        if classify_tool_calls(calls, self._ours):
            return "suppress"
        # Partition rule (mirror of the non-stream loop): suppress for the
        # reject round only when every call is ours or a hallucinated
        # infinitum_* name; `not classified` already proves >=1 rejectable.
        if self._guard and calls and all(
            (name := call.get("function", {}).get("name")) in self._ours
            or is_rejectable_memory_name(name, self._ours, self._client_names)
            for call in calls
        ):
            return "suppress"
        return "replay"

    def _scan_line(self, line: bytes) -> None:
        text = line.rstrip(b"\r").decode("utf-8", errors="replace").strip()
        if not text.startswith("data:"):
            return
        payload = text[5:].strip()
        if not payload or payload == "[DONE]":
            return
        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            return
        choices = data.get("choices") or []
        if not choices or not isinstance(choices[0], dict):
            return
        choice = choices[0]
        delta = choice.get("delta") or {}
        content = delta.get("content")
        if isinstance(content, str) and content:
            self._content_seen = True
        for call in delta.get("tool_calls") or []:
            if not isinstance(call, dict):
                continue
            function = call.get("function") or {}
            name = function.get("name")
            flat: dict[str, Any] = {
                "id": call.get("id"),
                "name": name,
                "arguments": function.get("arguments"),
            }
            if call.get("index") is not None:
                flat["index"] = call["index"]
            self._chunks.append(flat)
            if isinstance(name, str) and name and name not in self._ours:
                if not (
                    self._guard
                    and is_rejectable_memory_name(name, self._ours, self._client_names)
                ):
                    self._foreign_seen = True
        for field in self._reasoning_fields:
            value = delta.get(field)
            if isinstance(value, str) and value:
                self._reasoning_seen = True
                break
        reason = choice.get("finish_reason")
        if isinstance(reason, str) and reason:
            self._finish_reason = reason



    @property
    def content_seen(self) -> bool:
        return self._content_seen

    @property
    def forwarded(self) -> bool:
        """True once any byte of this round has been forwarded to the client."""
        return self._fwd_pos > 0

    @property
    def calls(self) -> list[dict[str, Any]]:
        """Flattened per-chunk tool_call deltas for reassemble_stream_tool_calls."""
        return self._chunks

    @property
    def finish_reason(self) -> str | None:
        return self._finish_reason

    def collected_bytes(self) -> bytes:
        return bytes(self._collected)

def _freeze_trigger(line: bytes) -> bool:
    """True for a complete SSE line tee-mode forwarding must stop before.

    Triggers: delta carries tool_calls, a non-null finish_reason, [DONE], or a
    data line that fails JSON parsing. Non-data (blank/comment) lines never
    trigger; an unparseable-but-not-prefixed line is not a data line at all.
    """
    text = line.rstrip(b"\r").decode("utf-8", errors="replace").strip()
    if not text.startswith("data:"):
        return False
    payload = text[5:].strip()
    if payload == "[DONE]":
        return True
    if not payload:
        return False
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        return True
    if not isinstance(data, dict):
        return True
    choices = data.get("choices") or []
    if not choices or not isinstance(choices[0], dict):
        return False
    choice = choices[0]
    delta = choice.get("delta") or {}
    if delta.get("tool_calls"):
        return True
    return choice.get("finish_reason") is not None


class ToolRuntime(Protocol):
    """Duck-typed view of the runtime the executors need (avoids import cycles)."""

    @property
    def retriever(self) -> Any: ...

    @property
    def db(self) -> Any: ...


def client_tool_names(tools: list[dict] | None) -> set[str]:
    """Collect the `function.name` strings a client request defines.

    Malformed entries (non-dicts, missing keys, non-string names) are ignored.
    """
    names: set[str] = set()
    if isinstance(tools, list):
        for tool in tools:
            try:
                name = tool["function"]["name"]
            except (TypeError, KeyError, IndexError):
                continue
            if isinstance(name, str):
                names.add(name)
    return names


def injected_tool_names(body_tools: Any) -> list[str]:
    """Return our tool names safe to inject (not already defined by the client).

    Malformed entries in the client's tools list are ignored. We never overwrite
    or shadow a client tool of the same name.
    """
    client_names = client_tool_names(body_tools)
    return [name for name in TOOL_NAMES if name not in client_names]


def is_rejectable_memory_name(
    name: object, ours: set[str], client_names: set[str]
) -> bool:
    """True for hallucinated Infinitum tool names: our prefix, not exposed
    this round, not defined by the client. Nameless/None calls never reject."""
    return (
        isinstance(name, str)
        and name.lower().startswith("infinitum_")
        and name not in ours
        and name not in client_names
    )


def build_reject_result(name: str, exposed: list[str]) -> str:
    """Tool-result JSON telling the model the memory tool name does not exist."""
    return json.dumps(
        {
            "error": f"unknown memory tool '{name}'",
            "available_memory_tools": exposed,
            "hint": "answer from the results above, or call one of these tools",
        },
        ensure_ascii=False,
    )


def build_tool_defs(names: Sequence[str]) -> list[dict[str, Any]]:
    """Return TOOL_DEFS filtered to `names`, in canonical TOOL_NAMES order."""
    wanted = set(names)
    return [TOOL_DEFS[name] for name in TOOL_NAMES if name in wanted]


def classify_tool_calls(calls: list[dict], ours: set[str]) -> bool:
    """Return True iff every call belongs to us (run the server-side loop).

    Single source of truth for the suppression/forward predicate on both stream
    and non-stream paths. Empty calls (missing/empty upstream choices) are
    terminal. Never raises.
    """
    if not calls:
        return False
    return all(str(c.get("function", {}).get("name", "")) in ours for c in calls)


def reassemble_stream_tool_calls(chunks: list[dict]) -> list[dict]:
    """Merge SSE tool_call_chunks into OpenAI message-shaped tool_call dicts.

    Primary strategy is index-merge: accumulate arguments per `index`, taking
    the first non-empty id/name per index. Fallback for index-less servers: a
    chunk with a non-empty id starts a new call, otherwise arguments append to
    the open call (heuristic, not spec). Callers building the assistant message
    MUST set `"role": "assistant"` explicitly; SSE deltas don't carry role.
    """
    if any("index" in chunk for chunk in chunks):
        by_index: dict[int, dict[str, Any]] = {}
        for chunk in chunks:
            idx = int(chunk.get("index", 0))
            slot = by_index.setdefault(idx, {"id": None, "name": None, "arguments": ""})
            if chunk.get("id") and not slot["id"]:
                slot["id"] = chunk["id"]
            if chunk.get("name") and not slot["name"]:
                slot["name"] = chunk["name"]
            slot["arguments"] += chunk.get("arguments") or ""
        return [
            {
                "id": slot["id"],
                "type": "function",
                "function": {"name": slot["name"], "arguments": slot["arguments"]},
            }
            for slot in (by_index[i] for i in sorted(by_index))
        ]
    calls: list[dict[str, Any]] = []
    for chunk in chunks:
        if chunk.get("id"):
            arguments = chunk.get("arguments") or ""
            calls.append(
                {
                    "id": chunk["id"],
                    "type": "function",
                    "function": {"name": chunk.get("name"), "arguments": arguments},
                }
            )
        elif calls:
            calls[-1]["function"]["arguments"] += chunk.get("arguments") or ""
    return calls


def _cap_payload(payload: Any, items_key: str = "results") -> str:
    """Serialize `payload` to JSON, trimming to MAX_RESULT_CHARS when over.

    List payloads: trim list length first, then entry content fields. Dict
    payloads (e.g. memory_get): trim the top-level content field until the JSON
    fits, so the result stays valid JSON. Sets top-level truncated=true whenever
    anything was cut; the final hard slice is an absolute last resort.
    """
    s = json.dumps(payload, ensure_ascii=False, default=str)
    if len(s) <= MAX_RESULT_CHARS:
        return s
    payload["truncated"] = True

    def size() -> int:
        return len(json.dumps(payload, ensure_ascii=False, default=str))

    items = payload.get(items_key)
    if isinstance(items, list):
        # Trim whole list entries first, then each entry's content field.
        while items and size() > MAX_RESULT_CHARS:
            items.pop()
        while items and size() > MAX_RESULT_CHARS:
            for item in items:
                content = str(item.get("content", ""))
                item["content"] = content[: max(0, len(content) - 256)]
    elif isinstance(payload.get("content"), str):
        # Trim this dict's content by the excess (plus JSON-escape margin) so
        # the serialized form shrinks below the cap instead of being sliced.
        while size() > MAX_RESULT_CHARS:
            content = str(payload["content"])
            excess = size() - MAX_RESULT_CHARS + 16
            if not content or excess <= 0:
                break
            payload["content"] = content[: max(0, len(content) - excess)]
    return json.dumps(payload, ensure_ascii=False, default=str)[:MAX_RESULT_CHARS]


async def execute(
    name: str,
    arguments_json: str,
    runtime: ToolRuntime,
    request_context: Any,
) -> str:
    """Run one memory tool call and return the tool-result JSON string.

    Never raises for model-caused errors (bad JSON, unknown id, missing keys);
    those become error-JSON strings the model can read. `request_context` is
    passed through to retriever.search unchanged and may be None.
    """
    try:
        args = json.loads(arguments_json) if arguments_json else {}
    except json.JSONDecodeError:
        return '{"error": "invalid arguments"}'
    if not isinstance(args, dict):
        return '{"error": "invalid arguments"}'

    if name == "infinitum_memory_search":
        query = str(args.get("query", ""))[:512]
        try:
            limit = min(int(args.get("limit", 10) or 10), MAX_SEARCH_LIMIT)
        except (TypeError, ValueError):
            limit = 10  # non-numeric model-provided limit; fall back to the default
        scored = await runtime.retriever.search(query, limit=limit, request_context=request_context)
        payload: dict[str, Any] = {
            "results": [
                {
                    "id": sm.memory.id,
                    "memory_type": sm.memory.memory_type,
                    "topic": sm.memory.topic,
                    # search only returns active memories; kept for provenance honesty
                    "status": sm.memory.status,
                    "score": round(sm.score, 3),
                    "content": sm.memory.content,
                }
                for sm in scored
            ]
        }
        return _cap_payload(payload)

    if name == "infinitum_memory_get":
        memory = await runtime.db.get_memory(str(args.get("memory_id", "")))
        if memory is None:
            return '{"error": "not found"}'
        return _cap_payload(memory.model_dump(mode="json"), items_key="")

    return '{"error": "unknown tool"}'

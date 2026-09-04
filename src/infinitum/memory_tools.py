"""Read-only memory drill-down tools exposed to the model when tools_enabled.

These executors never mutate state: `retriever.search` and `db.get_memory` are
read-only. Model-caused errors return error-JSON strings instead of raising so
the server-side tool loop can feed them back to the model.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
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


class ToolRuntime(Protocol):
    """Duck-typed view of the runtime the executors need (avoids import cycles)."""

    @property
    def retriever(self) -> Any: ...

    @property
    def db(self) -> Any: ...


def injected_tool_names(body_tools: Any) -> list[str]:
    """Return our tool names safe to inject (not already defined by the client).

    Malformed entries in the client's tools list are ignored. We never overwrite
    or shadow a client tool of the same name.
    """
    client_names: set[str] = set()
    if isinstance(body_tools, list):
        for tool in body_tools:
            try:
                name = tool["function"]["name"]
            except (TypeError, KeyError, IndexError):
                continue
            if isinstance(name, str):
                client_names.add(name)
    return [name for name in TOOL_NAMES if name not in client_names]


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

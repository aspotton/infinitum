# API reference

The HTTP surface, request-control headers, and the server-side memory-tool contract. Configuration keys live in [./CONFIGURATION.md](./CONFIGURATION.md); mechanism internals in [./ARCHITECTURE.md](./ARCHITECTURE.md).

## OpenAI-compatible endpoints

### `POST /v1/chat/completions`

Supports normal and streaming Chat Completions. Unknown request fields are preserved and forwarded upstream.

### `GET /v1/models`

Transparent upstream passthrough.

## Runtime endpoints

### `GET /health`

Returns service and feature status.

### `GET /memory`

Lists memories. Optional `status` query parameter.

### `POST /memory`

Manually inserts a memory.

```json
{
  "memory_type": "goal",
  "topic": "runtime",
  "content": "Build an effective persistent memory layer for LLMs.",
  "importance": 1.0,
  "confidence": 1.0
}
```

### `POST /memory/search`

```json
{
  "query": "database choice",
  "limit": 20
}
```

Optional temporal parameters:

```json
{
  "query": "database choice",
  "temporal_view": "as_of",
  "as_of": "2024-03-15"
}
```

`temporal_view` selects the validity window applied before scoring: `"current"` (default; rows whose `valid_until` is already past are dropped), `"all"` (no temporal filtering), or `"as_of"` (rows whose validity window covers `as_of`, an ISO date or datetime; a date-only value counts as 00:00:00 UTC that day). An `"as_of"` view without an `as_of` value, an unparseable `as_of`, or an unknown view string returns HTTP 400.

### `GET /memory/{id}`

Returns memory plus provenance event IDs. The response also carries an `observations` array: the memory's first-class evidence records, ordered by `observed_at` (fields: `id`, `memory_id`, `observed_at`, `session_id`, `evidence_type`, `evidence_weight`, `confidence`, `fingerprint`, `metadata_json`, `source_event_ids`). Memories predating the observations ledger carry weight-0.5 `legacy` rows derived from their source events.

### `DELETE /memory/{id}`

Archives the memory rather than destroying historical events.

### `GET /events`

Inspects immutable events. `session_id`, `user_id`, and `project_id` can be supplied to filter.

### `GET /request-context`

Returns the user/project/CWD context resolved from the current request headers. This is a diagnostic endpoint and does not authenticate the caller.

### `GET /topics`

Lists generated multi-memory topic summaries.

## Per-request control headers

Internal headers are stripped before forwarding upstream.

```text
X-Infinitum-Memory: off
X-Infinitum-Learning: off
X-Infinitum-Session-ID: my-session-id
# Also accepted: X-OpenCode-Session / X-Session-Id / X-Session-Affinity
X-Infinitum-User-ID: adam
X-Infinitum-Project-ID: infinitum
X-Infinitum-CWD: /home/adam/infinitum
X-Infinitum-Debug: true
```

`X-Infinitum-Debug: true` adds response metadata such as the number of detailed memories injected, token budget used, resolved user/project IDs, and whether project identity was derived from CWD. It intentionally does not return the full CWD in response headers.

## Memory drill-down tools

The compiler injects a bounded memory block per request. When the model needs more than that block shows, optional read-only tools let it drill deeper. The tools are enabled via `memory.tools_enabled`; see [CONFIGURATION](./CONFIGURATION.md).

When enabled, Infinitum appends two function tools to the client's tool list on every memory-enabled request, as long as the client has not defined a tool of the same name:

- `infinitum_memory_search(query, limit)`, a ranked search over active memories (default limit 10, max 50);
- `infinitum_memory_get(memory_id)`, full current content plus provenance source event ids for one memory.

The tool definitions are exposed statically on memory-enabled requests unless the client defines tools with the same names, so the tools region never flickers between turns.

The tools use the same scorer as injection, so results are ranked identically. They read the same global namespace, and request user/project/CWD context stays a soft affinity, never an access-control filter.

### Transparency

The tool loop runs server-side and is transparent to the client: intermediate tool-call rounds never appear in the client's response or stream, and the client sees only the final answer. The one caveat is plain model text on streaming responses: under the default `memory.stream_reasoning: live`, a round's visible output can reach the client as it streams, including reasoning deltas and even a plain-text preamble the model emits alongside an internal memory lookup, while that lookup itself runs server-side. Infinitum's own tool-call structures stay invisible even in a round where such text sits beside the call: visible text, invisible machinery. That transparency is unconditional for memory-named tool calls: a hallucinated `infinitum_*` call is answered server-side and never forwarded, per the parser-hazard note below.

### Rounds, forcing, and synthesis

Up to 4 tool rounds run per request. If the model keeps calling the memory tools past that cap, Infinitum makes one final forced answer round: its own tool definitions are removed and the standard `tool_choice: "none"` parameter is sent, so OpenAI-compatible servers with automatic tool-call parsers must answer in text instead of re-emitting the tool calls, and the server-side transcript carries an explicit instruction to answer now from the gathered tool results, so a model that was mid-planning cannot end the turn with a dangling "Let me check..." line.

If a server ignores `tool_choice` and the forced round still comes back blank, both the streaming and non-streaming paths synthesize an assistant answer from the already-gathered tool results, leading with `Based on the retrieved memories:`, rather than forwarding a null-content message or a dangling tool call. A tool loop that runs out of rounds with every round suppressed ends the same way, with a synthesized answer instead of silence.

### Event recording

Suppressed rounds are recorded as `memory.tool_call` events, and rejected calls are recorded the same way with a `rejected` provenance flag plus the exact instructive result the model was told, and forward-stripped calls from a mixed terminal round are recorded alongside them with a `stripped` provenance flag and are never answered; only the final assistant message becomes an assistant event.

### Debug counters

With `X-Infinitum-Debug: true`, the response carries `x-infinitum-memory-tool-calls` with the round count and, when calls were rejected or forward-stripped, `x-infinitum-memory-tool-rejects` with the combined count. On streaming responses these two counters instead arrive as SSE comment lines after the final `[DONE]` in the stream body; comment lines are ignored by SDK parsers that stop reading at `[DONE]`, so treat them as proxy- and log-visible. Non-stream responses keep the headers.

## Latency edges

Streaming reasoning is controlled by `memory.stream_reasoning`. In the default `live` mode, model reasoning deltas stream out immediately while the rest of a round is held only until Infinitum can tell whether it is a memory tool call; `buffered` holds everything back until that decision. Tool calls and results are never forwarded in either mode, though a round's plain text and thinking may be visible in the stream under `live`. With memory tools enabled, a round's plain text no longer short-circuits that decision, since the round is classified on its tool-call structure, so in `buffered` mode an answer round is held until the upstream finishes generating it (a slower first token there, the price of not leaking tool bytes); the default `live` mode streams answer content incrementally and is unaffected.

Each tool round is a full extra upstream round-trip. A long loop multiplies upstream latency and a client-side timeout can fire mid-loop.

If your upstream rejects or mishandles tool definitions, turn the flag off; requests then behave exactly as before.

## Parser-hazard note

Some upstreams run automatic tool-call parsers and emit tool calls on their own. A call to an `infinitum_*` name that Infinitum did not expose this request is rejected server-side: the model receives an instructive tool result listing the memory tools that actually exist, and the loop continues without the client seeing anything. That covers names Infinitum never had, such as `infinitum_retrieve`, and also Infinitum's own names when an upstream prompt-cache diff drops the tool definitions mid-conversation. Any other foreign name stays terminal and is forwarded to the client, except that any `infinitum_`-prefixed call the client did not define is dropped from the forwarded response, so the client sees only its own tool calls: visible text, invisible machinery now holds for mixed rounds too, where a memory lookup sits beside a client tool call. Names the client defines belong to the client: a client tool is forwarded even if its name starts with `infinitum_`.

See [../README.md](../README.md) for the product overview and [./ARCHITECTURE.md](./ARCHITECTURE.md) for how these behaviors are implemented.

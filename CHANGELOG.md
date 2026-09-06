# Changelog

## Unreleased

- Added `memory.stream_reasoning` (default `live`): streaming responses now pass a thinking model's reasoning deltas through to the client while a round is still undecided, instead of holding every token until the round's purpose is known; set `buffered` to restore the prior hold-everything behavior.
- Added `memory.reasoning_delta_fields` to name the SSE delta fields that carry model reasoning for the live mode.
- Memory tool calls and tool results are still never forwarded in either mode, and a live stream freezes the moment a tool-call line appears; under `live`, thinking text from intermediate tool rounds may now be visible in the client stream.
- Streaming tool rounds after the first now run inside the streamed response; an upstream failure there surfaces either verbatim with zero bytes already sent, or as an in-stream SSE error event once bytes are committed, rather than failing a response whose status is already fixed.
- Streaming `X-Infinitum-Debug` round counters now ride the stream as trailing SSE comment lines after `[DONE]` instead of response headers; SDK parsers that stop at the sentinel never see them, and non-stream responses keep their debug headers unchanged.
- Added regression tests for the live and buffered modes, byte parity, reasoning passthrough, and error surfacing; the full suite now covers 181 tests.

## 0.2.7

- Retrieval eligibility now requires at least one genuine relevance signal (semantic, lexical, or topic) above `memory.minimum_relevance_score` (default `0.08`; set `0.0` to disable) before importance, confidence, or freshness can qualify a memory; the high-importance goal/decision exemption is unchanged and the drill-down memory tools and `POST /memory/search` share the same gated scorer.
- A topic-summary refresh that regenerates an identical summary no longer writes the topic row, so it no longer bumps the cache-invalidation watermark that pins compiled context; memory-write bumps still move the watermark by design.
- Recorded user/assistant text, learn-job payloads, and `memory.tool_call` metadata now strip echoed `<infinitum_memory>` regions, so quoted context blocks can never be re-learned into memory, while the `request.received` audit event stays byte-exact with what the client sent.
- Added regression tests for the relevance gate, the identical-summary watermark guard, and echo sanitization; the full suite now covers 163 tests.

## 0.2.6

- Added `learning.skip_when_upstream_busy` (default `false`): the learning worker no longer claims new jobs while Infinitum is actively proxying a request to the upstream.
- Added `learning.upstream_idle_grace_seconds` (default `0`): keeps learning deferred until the upstream has been continuously idle for that long after traffic drains; any new foreground request restarts the window.
- Deferral happens only at job boundaries — an in-flight extraction or topic-summary operation is never interrupted.
- Deferred learning work stays in the durable job queue, so nothing is lost while the worker waits for the upstream to go idle.
- `GET /health` now reports a live `active_requests` count showing why the learning worker is idle.
- Both options default to off, so unset configurations keep byte-identical prior behavior.
- Added regression tests for the defer/drain paths and idle-grace windows; the full suite now covers 135 tests.

## 0.2.5

- Fixed a cache-pollution bug in session pinning: headerless requests previously generated a fresh session id per call, never hit the pin cache, and evicted real sessions' pins from the 64-entry cache.
- A generated session id is now used only for event provenance; headerless requests compile fresh every time without touching the cache.
- Clients that send `X-Infinitum-Session-ID` (or an alias/metadata session id) are byte-for-byte unaffected.
- Memory injection itself is unchanged for every client; only the pin cache's bookkeeping changed.

## 0.2.4

- Added a reject-and-instruct guard for hallucinated memory-tool calls: an `infinitum_*` call that Infinitum did not expose and the client did not define is rejected server-side with an instructive tool result naming the two real memory tools, and the tool loop continues.
- Covers model-invented names such as `infinitum_retrieve`, and Infinitum's own names when an upstream prompt-cache diff drops the tool definitions mid-conversation.
- Names the client defines are always still forwarded, even when they start with `infinitum_`.
- Streaming never leaks the hallucinated name's bytes to the client; the non-streaming tool loop rejects the call before anything is forwarded.
- Model-facing wording now states that the memory tool set is complete and exclusive.
- With `X-Infinitum-Debug: true`, responses gain `x-infinitum-memory-tool-rejects` alongside the existing call counter.
- The guard is fully dormant when memory tools are off.

## 0.2.3

- Made the compiled memory block cache-stable for upstream prompt caching: the block is session-pinned and byte-identical across turns until memory or topic state changes, tracked by an invalidation watermark over memories and topics.
- Added `context.inject_position` config, default `suffix`, placing the memory message immediately before the last user message; set `prefix` for strict chat templates that require the system message at index 0.
- The two drill-down tool definitions are now exposed statically on every memory-enabled request when `memory.tools_enabled` is on (unless the client defines the same names), so the tools region never flickers.
- Retrieval ranking breaks ties on memory id for deterministic ordering at equal scores.
- The memory tool loop now forces a final answer round with Infinitum's tool definitions removed after its 4-round cap.

## 0.2.2

- Fixed lost memory-learning turns when an OpenAI-compatible server returns `finish_reason="tool_calls"` with empty assistant content.
- Memory extraction now recovers schema-shaped `message.tool_calls[].function.arguments` or legacy `function_call.arguments` when they contain the expected memory JSON; unrelated tool calls are ignored.
- Topic summarization can recover a simple `summary`, `text`, or `content` value from tool-call arguments before using the deterministic fallback.
- Extraction and summary prompts explicitly instruct background models not to call tools/functions.
- Empty-output diagnostics now include tool-call count and function names.
- Documented `learning.extra_body.tool_choice: none` as the preferred defense for servers with aggressive automatic tool-call parsing; this composes with `chat_template_kwargs.enable_thinking: false`.
- Added regression tests for schema-shaped tool-call extraction, unrelated tool-call rejection, and tool-call topic-summary recovery.

## 0.2.1

- Fixed repeated topic-summary job failures when an OpenAI-compatible learning model returns an empty final `message.content`.
- Empty topic-summary output now degrades to a bounded deterministic summary of active canonical memories instead of retrying the same doomed job up to `learning.max_attempts` times.
- Added non-content diagnostics for empty learning responses (`finish_reason`, reasoning character count, and completion-token usage when available).
- Added `learning.extra_body` for vendor-specific OpenAI-compatible background-learning controls. This can be used with some OpenAI-compatible servers that expose chat-template controls, to disable reasoning/thinking for memory extraction and topic summarization.
- Added `learning.topic_summary_fallback_memories` to bound deterministic fallback summaries.
- On startup, dirty topic state left behind by an older failed summary job is requeued automatically when its learning model can be recovered, so upgrading fixes existing failed topics without waiting for a new interaction.
- Memory extraction now uses the same robust assistant-content parser as normal non-streaming responses and treats empty final content as a logged no-op rather than a retry storm.
- Added regression tests for empty reasoning-model topic output and background-learning request extensions.

## 0.2.0

- Renamed the project to **Infinitum**.
- Renamed the primary Python distribution/package to `infinitum`.
- Renamed the primary CLI to `infinitum`.
- Added compatibility shims for the pre-0.2 `context_runtime` Python namespace and `context-runtime` CLI.
- Added canonical `INFINITUM_CONFIG` configuration environment variable while retaining `CONTEXT_RUNTIME_CONFIG` as a fallback.
- Added canonical `x-infinitum-*` request-context/control headers while retaining pre-0.2 `x-context-*` aliases.
- Renamed the injected memory envelope to `<infinitum_memory>`.
- Changed the default database filename for new installations to `infinitum.db`.
- Added automatic reuse of an existing `./context-runtime.db` when no database path is explicitly configured and no `./infinitum.db` exists.
- Updated README, architecture, roadmap, examples, Docker configuration, tests, and agent-development guidance for the Infinitum name.
- Preserved all v0.1.4 memory, request-context, semantic-reinforcement, and incremental-topic behavior.

## 0.1.4

- Added a first-class request-context resolver for user, project, and CWD hints.
- Added canonical `x-context-user-id`, `x-context-project-id`, and `x-context-cwd` support plus configurable OpenCode, Headroom, and LiteLLM-compatible aliases, including OpenCode's `x-opencode-directory`.
- Added OpenCode-compatible session header resolution from `x-opencode-session`, `x-session-id`, and `x-session-affinity`.
- Added deterministic project-ID derivation from normalized CWD when no explicit project ID is supplied.
- Persisted request context on `events` and `requests`; existing databases migrate in place with nullable columns.
- Carried request context into asynchronous learning jobs and stored learned-memory origin/reinforcement context metadata.
- Added soft same-user / same-project / same-CWD retrieval affinity after the normal relevance threshold; memory remains globally visible.
- Added per-memory affinity diagnostics to search results.
- Stripped consumed identity/context aliases before ordinary upstream forwarding.
- Added optional canonical `x-headroom-*` forwarding from resolved context for Infinitum -> Headroom deployments.
- Added `GET /request-context` diagnostics and user/project filters to `GET /events`.
- Added debug response headers for resolved user/project identity and CWD-derived project status.
- Added migration, header-resolution, affinity, upstream-header, and API provenance tests.

## 0.1.3

- Replaced lexical-only reinforcement with guarded lexical + semantic equivalence.
- Reinforcement now requires exact memory type and topic compatibility.
- Added learner-proposed `reinforces_memory_id` targeting with deterministic validation.
- Explicit corrections/supersession can no longer be accidentally reinforced due to similar wording.
- Made reinforcement idempotent for already-attached source interactions so retries do not inflate `observation_count`.
- Added reinforcement decision metadata (method and match scores) for debugging/audit.
- Fixed explicit `MemoryRetriever.search(..., limit=N)` so it actually returns at most `N` results; this bounds the learning prompt to the intended nearby-memory count.
- Added tests for semantic reinforcement, type/topic guards, targeted reinforce hints, supersession precedence, idempotent observation counting, and retrieval limits.
- Expanded the roadmap with first-class `memory_observations` / evidence weighting and independent-source semantics.

## 0.1.2

- Replaced per-interaction full-topic summary regeneration with incremental topic-summary maintenance.
- Added durable `topic_updates` dirty state.
- Added coalescing/debounce and immediate threshold-triggered summary scheduling.
- Existing summaries now update from changed records plus a bounded context sample.
- Initial topic summaries bootstrap from a bounded recent active-memory sample.
- Dirty revisions are cleared only after successful summarization and are revision-safe if the same memory changes while a summary is running.
- Added dedicated topic-summary generation token cap.
- Added tests for coalescing and bounded incremental summary prompts.
- Expanded architecture/roadmap around three processing timescales: every interaction, incremental, and periodic deep consolidation.

## 0.1.1

- Added a separate background-learning timeout and bounded learning completion tokens.
- Improved learning timeout diagnostics.

## 0.1.0

- Initial global-memory OpenAI-compatible proxy prototype.

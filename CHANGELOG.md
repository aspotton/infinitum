# Changelog

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
- Added `learning.extra_body` for vendor-specific OpenAI-compatible background-learning controls. This can be used with servers such as vLLM/Qwen to disable reasoning/thinking for memory extraction and topic summarization.
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

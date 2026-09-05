# AGENTS.md — Infinitum

This file is the working guide for coding agents contributing to Infinitum.

## Project identity

- Product/repository name: **Infinitum**
- Tagline: **Persistent memory and context for AI.**
- Repository slug: `infinitum`
- Python distribution/package: `infinitum`
- Primary CLI: `infinitum`
- Current release line: `0.2.x`

The pre-0.2 project name was **Context Runtime**. Compatibility shims remain for the old `context_runtime` Python namespace, `context-runtime` CLI, `CONTEXT_RUNTIME_CONFIG` environment variable, and `X-Context-*` HTTP headers. Do not use the old name in new public APIs, examples, or prose except when documenting migration/compatibility.

## Purpose

Infinitum is an OpenAI-compatible memory and context runtime. A client points an ordinary OpenAI SDK or agent framework at Infinitum; Infinitum retrieves useful persistent memory, compiles a bounded context block, forwards the request to a configurable OpenAI-compatible upstream, records immutable interaction events, and learns durable memories asynchronously.

The current implementation intentionally has one globally visible memory namespace. User/project/CWD request context is provenance plus soft retrieval affinity, not authentication or isolation.

## Architectural invariants

Preserve these unless a deliberate architecture change is documented and tested:

1. **Events are truth; memories are derived.** Never destroy raw evidence merely because derived memory changes.
2. **LLMs propose memory changes; deterministic code mutates state.** Extraction output must not directly rewrite the database.
3. **Current state may supersede history without deleting history.** Supersession is a derived-state transition.
4. **Reinforcement is stricter than retrieval.** Broad retrieval is acceptable; merging two memories requires type/topic compatibility plus credible lexical/semantic evidence.
5. **Learning stays off the foreground response path.** A slow or failed learner must not block a successful user response.
6. **Memory processing has three timescales:** every interaction, incremental topic maintenance, periodic deep consolidation (roadmap).
7. **Context size is a ceiling, not a target.** Prefer value per token over filling the model window.
8. **Provenance is mandatory.** A derived memory should remain traceable to source events.
9. **Soft request context is not security.** Until scoped memory is implemented, never imply that `user_id` or `project_id` headers isolate data.
10. **Future scoped retrieval must filter eligibility before semantic ranking.** Do not retrieve globally and filter unauthorized results afterward.

## Package layout

Primary code lives in `src/infinitum/`.

- `runtime.py` — DI hub: `build_runtime()` constructs Database, EmbeddingClient, UpstreamClient, Retriever, Compiler, Learner, RequestContextResolver, TokenCounter; also runs startup dirty-topic recovery. Routes access this via `app.state.runtime`.
- `app.py` — thin FastAPI construction/lifespan (starts/stops the learning worker)
- `__main__.py` — `infinitum` CLI entrypoint (argparse + `INFINITUM_CONFIG` + uvicorn factory)
- `routes/openai.py` — OpenAI-compatible proxy endpoints and per-request controls
- `routes/memory.py` — memory management/search endpoints
- `routes/admin.py` — health, event, topic, and request-context diagnostics
- `config.py` — configuration models and loading (incl. legacy-DB reuse)
- `database.py` — SQLite schema, persistence, durable jobs, provenance (largest module)
- `request_context.py` — user/project/CWD header resolution
- `retrieval.py` — hybrid scoring and context affinity
- `compiler.py` — token-aware memory selection/rendering/injection
- `learning.py` — extraction, reinforcement, supersession, incremental topic summaries, worker
- `text.py` — shared scoring primitives: normalization, lexical/topic similarity, freshness decay (used by retrieval and reinforcement guards)
- `tokenizer.py` — `TokenCounter` feeding the compiler token budget
- `embeddings.py` — OpenAI-compatible embedding client
- `upstream.py` — transparent OpenAI-compatible upstream transport
- `models.py` — event/memory/request-context models

`src/context_runtime/` is compatibility-only. Do not add new behavior there; implement it in `src/infinitum/` and expose a wrapper only if old imports need to keep working.

## Public naming and compatibility

Use these canonical names in new code/docs:

- `INFINITUM_CONFIG`
- `X-Infinitum-User-ID`
- `X-Infinitum-Project-ID`
- `X-Infinitum-CWD`
- `X-Infinitum-Session-ID`
- `X-Infinitum-Memory`
- `X-Infinitum-Learning`
- `X-Infinitum-Debug`
- `<infinitum_memory>...</infinitum_memory>`
- `infinitum_memory_search` / `infinitum_memory_get` (read-only drill-down tool names, not headers; gated by `memory.tools_enabled`)
- `x-infinitum-memory-tool-rejects` (debug-only response header; count of rejected hallucinated memory-tool calls)

A tool call whose name starts with `infinitum_` but was not exposed this request and is not client-defined is a hallucination: reject it server-side with an instructive tool result, never forward it to the client, and let the loop continue.

Legacy `X-Context-*` headers remain accepted but should be lower priority than canonical Infinitum headers. Runtime-only headers must be stripped before normal upstream forwarding. Headroom forwarding, when explicitly enabled, should be generated from the already-resolved request context rather than blindly forwarding inbound identity-like headers.

## Database compatibility

Existing v0.1.x SQLite databases must remain usable unless a migration is explicitly introduced. The old implicit filename was `context-runtime.db`; the new default is `infinitum.db`. `load_config()` intentionally reuses an existing legacy DB when no database path is explicitly configured and no new default DB exists.

Schema changes must be additive/migratable and must preserve immutable events and existing memory IDs unless a documented migration absolutely requires otherwise.

## Memory learning rules

Per interaction, the learner should see the current interaction plus a bounded nearby-memory set, not the entire corpus.

For candidate mutation:

- explicit correction/supersession takes precedence over reinforcement;
- reinforcement requires exact memory type/topic compatibility;
- near-identical lexical matches may reinforce deterministically;
- high semantic similarity may reinforce when embeddings are available;
- learner-proposed `reinforces_memory_id` is advisory and must pass deterministic guards;
- retries/replays of the same source events must not inflate `observation_count`.

Topic summaries are incremental: dirty topic deltas are coalesced, then an existing summary is updated from a bounded set of changed memories plus a small context sample. Do not regress to resending an entire large topic after every interaction.

An empty final response from a reasoning/local model must not create a retry storm. Topic summaries should degrade to a bounded deterministic active-memory representation; detailed memories remain authoritative. Vendor-specific background request knobs belong under `learning.extra_body`, never in foreground proxy requests.

Background learning may optionally defer job start while foreground proxy requests are in flight via `learning.skip_when_upstream_busy`; deferred work stays in the durable job queue (deferred, not dropped) and consumes no attempt count. Additionally, `learning.upstream_idle_grace_seconds` (default 0) keeps claims deferred until the upstream has been continuously idle for that long since the last foreground activity, with any new request restarting the window. The active-request counter lives in `runtime.py` and is instrumented only in `routes/openai.py` around proxied upstream calls; the learner's own upstream calls bypass it by design, so the worker cannot starve itself.

## Code map

Foreground path:

```
pyproject [project.scripts] → __main__.py:main → uvicorn(create_app)
app.py:create_app → runtime.py:build_runtime → app.state.runtime; worker.start()
routes/openai.py:chat_completions → request_context.resolve → compiler.compile
  → retrieval.search → compiler.inject → db.add_request_memory → upstream
  → _record_completion → db.add_event + db.enqueue_job("learn_interaction")
```

Background path:

```
learning.py:LearningWorker._run → db.claim_job (jobs table, SQLite)
  → MemoryLearner.learn → retriever.search (bounded nearby set)
  → LLM proposes → _apply (deterministic guards; text.py similarity) → db.mark_topic_dirty
  → refresh_topic_summary → _deterministic_topic_fallback on empty output
  → db.finish_job / db.fail_job (backoff 2**attempts, capped 60s)
```

Key symbols: `build_runtime` (runtime.py), `create_app` (app.py), `chat_completions` (routes/openai.py), `ContextCompiler.compile/inject` (compiler.py), `MemoryRetriever.search` (retrieval.py), `MemoryLearner.learn/_apply` (learning.py), `LearningWorker` (learning.py), durable job queue `enqueue_job/claim_job/finish_job/fail_job` (database.py).

## Testing

Run from the repository root:

```bash
pytest -q
```

For style/static checks when Ruff is installed:

```bash
ruff check .
```

Any change to request headers, database migration, reinforcement, retrieval limits, streaming, or background learning should include or update focused tests.

Test conventions: flat `tests/` (12 files, no conftest, no shared helpers). DB isolation via `tempfile.TemporaryDirectory()`; upstream faked either by `httpx.MockTransport` handler (foreground proxy) or `AsyncMock` on `learning_chat_completion`/`retriever.search` (background). Tests construct `AppConfig()` directly, mutate fields, call `create_app(cfg)`, and wrap in `TestClient`.

Before packaging a release, verify at minimum:

- `import infinitum` works;
- the `infinitum` CLI resolves;
- legacy `import context_runtime` still works during the compatibility period;
- canonical `X-Infinitum-*` headers win over legacy aliases;
- legacy `X-Context-*` controls are still accepted;
- runtime-only headers do not leak upstream;
- an existing v0.1.x DB can be opened without losing data;
- all tests pass.

## Roadmap direction

Read `docs/ROADMAP.md` before implementing larger features. Key future work includes:

- benchmark/evaluation harness for extraction and retrieval quality;
- first-class `memory_observations` and evidence weighting;
- periodic deep consolidation and canonicalization;
- hard user/project/session/agent memory scopes;
- authenticated identity from a trusted LiteLLM edge;
- organization/team memory and authoritative directives/goals;
- document/source expansion and progressive retrieval tools;
- PostgreSQL/pgvector and distributed operation;
- Responses API support;
- observability and admin/inspection tooling.

The long-term goal is not merely to store more history. It is to give otherwise stateless models coherent continuity while keeping memory current, explainable, bounded, and replaceable as models improve.

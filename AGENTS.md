# AGENTS.md — Infinitum

This file is the working guide for coding agents contributing to Infinitum.

## Project identity

- Product/repository name: **Infinitum**
- Tagline: **Persistent memory and context for AI.**
- Repository slug: `infinitum`
- Python distribution/package: `infinitum`
- Primary CLI: `infinitum`
- Current release line: `0.2.x`

The pre-0.2 project name was **Context Runtime**. All pre-0.2 compatibility shims (the old Python namespace, the old CLI alias, the old config environment variable, and the old header aliases) were removed. Do not use the old name in new public APIs, examples, or prose except in historical notes.

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
- `config.py` — configuration models and loading
- `database.py` — SQLite schema, persistence, durable jobs, provenance, `memory_observations`/`memory_observation_sources` evidence ledger, and additive temporal columns on `memories` (largest module)
- `request_context.py` — user/project/CWD header resolution
- `retrieval.py` — hybrid scoring and context affinity
- `compiler.py` — token-aware memory selection/rendering/injection
- `learning.py` — extraction, reinforcement, supersession, incremental topic summaries, worker
- `text.py` — shared scoring primitives: normalization, lexical/topic similarity, bounded phrase comparison (length-bound skip + 8192-char backstop), freshness decay (used by retrieval and reinforcement guards)
- `tokenizer.py` — `TokenCounter` feeding the compiler token budget
- `embeddings.py` — OpenAI-compatible embedding client
- `upstream.py` — transparent OpenAI-compatible upstream transport
- `models.py` — event/memory/request-context models

Dev-only evaluation code lives in `benchmarks/` (not an installed package; run everything from the repo root via `python -m benchmarks.run` / `python -m benchmarks.replay`; see `benchmarks/README.md`): a golden-scenario YAML corpus, a deterministic offline runner with a scripted extractor, extraction/retrieval precision-recall + token metrics, and live-replay against a running instance. For paraphrase-tolerant semantic grading of a **live** store against this corpus, use the `.opencode/skills/infinitum-eval` skill from an agent session in this repo.

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
- `x-infinitum-memory-tool-rejects` (debug-only response header; count of rejected or forward-stripped memory-tool calls)

A tool call whose name starts with `infinitum_` but was not exposed this request and is not client-defined is a hallucination: reject it server-side with an instructive tool result, never forward it to the client, and let the loop continue. On a mixed terminal round, where memory-namespace calls sit alongside client tool calls, non-client `infinitum_` calls are stripped from the forwarded response on both the stream and non-stream paths rather than forwarded, and are recorded as `memory.tool_call` events with `stripped: true` and never answered.

Runtime-only headers must be stripped before normal upstream forwarding. Headroom forwarding, when explicitly enabled, should be generated from the already-resolved request context rather than blindly forwarding inbound identity-like headers.

## Database compatibility

The default database filename is `infinitum.db`. There is no legacy-database auto-detection; when an older SQLite database is pointed at via an explicit `memory.database_path`, it is opened and its schema migrates in place.

Schema changes must be additive/migratable and must preserve immutable events and existing memory IDs unless a documented migration absolutely requires otherwise.

## Memory learning rules

Per interaction, the learner should see the current interaction plus a bounded nearby-memory set, not the entire corpus.

- Echoed `<infinitum_memory>` regions must be stripped from recorded user/assistant text, learn-job payloads, and `memory.tool_call` metadata; `request.received` stays byte-exact as received, by design (events are truth; message text is derived).

For candidate mutation:

- explicit correction/supersession takes precedence over reinforcement;
- an explicit correction may supersede across a drifted topic as a replacement of the target memory, not a merge; implicit supersede and reinforcement remain type/topic-gated (invariant 4 unchanged);
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

Streaming has two `memory.stream_reasoning` modes: `live` tees model reasoning deltas to the client while a round is undecided and freezes at the first tool-call line; `buffered` holds all round bytes until the decision. Round-1 decisions and pre-byte errors behave identically in both modes; live mode additionally tees round-1 reasoning deltas to the client before the decision, while buffered holds them. With memory tools on, a round classifies on its tool-call structure rather than on content alone: content co-resident with only-ours or rejectable calls suppresses and loops (visible text, invisible machinery), a foreign name stays terminal mid-stream, and a blank forced terminal or a cap-exhausted loop that suppressed every round ends with an answer synthesized from the gathered tool results on both the streaming and non-streaming paths. When the guard is active, mixed-round stripping is wired into every forwarded byte-yield site (accum replay, live tail, mid-round consume output, held-burst flush, and buffered/forced replays), so a forwarded stream carries no `infinitum_` bytes while `completed()` and the recorded events stay on raw bytes. Rounds 2+ run inside the returned `StreamingResponse`: upstream failures there are either a 0-byte verbatim error re-presentation (nothing forwarded yet) or an in-stream SSE error event (bytes already forwarded), never HTTP statuses; suppressed rounds' `[DONE]` sentinels are never forwarded. Streaming debug counters ride the stream as trailing SSE comments in both modes (headers stay on non-stream responses only). `completed()` sees the final round only.

Background path:

```
learning.py:LearningWorker._run → db.claim_job (jobs table, SQLite)
  → MemoryLearner.learn → retriever.search (bounded nearby set)
  → LLM proposes → _apply (deterministic guards; text.py similarity) → db.mark_topic_dirty
  → refresh_topic_summary → _deterministic_topic_fallback on empty output
  → db.finish_job / db.fail_job (backoff 2**attempts, capped 60s)
```

Interrupted jobs self-heal two ways: `claim_job` adopts a stale `running` lock past a lease of `learning.timeout_seconds + 60s` (derived from config, no new config key), and `recover_interrupted_jobs` in `build_runtime` requeues interrupted jobs at startup. The startup call sits above the dirty-topic recovery, though the order is functionally neutral. Both paths assume a single process: the startup requeue presumes no other live instance is using the DB.

Key symbols: `build_runtime` (runtime.py), `create_app` (app.py), `chat_completions` (routes/openai.py), `ContextCompiler.compile/inject` (compiler.py), `MemoryRetriever.search` (retrieval.py), `MemoryLearner.learn/_apply` (learning.py), `LearningWorker` (learning.py), durable job queue `enqueue_job/claim_job/finish_job/fail_job` (database.py).

## Release notes

All release changes, new-version descriptions, and feature narratives belong in `CHANGELOG.md`. The README keeps only the single `Current release: vX.Y.Z` line plus a pointer to the changelog — do not add or update version-release prose in README. Write entries in CHANGELOG.md's dash-bullet style and keep them vendor-neutral: describe server behaviors generically (for example, "OpenAI-compatible servers with automatic tool-call parsers") instead of naming specific vendor stacks.

## Versioning

- The single source of truth for the version is `__version__` in `src/infinitum/__init__.py`.
- `pyproject.toml` reads it dynamically via hatchling (`[tool.hatch.version]`); do not duplicate the number there.
- The FastAPI/OpenAPI version and `/health` derive from `__version__` in code; do not hardcode versions in routes or app construction.
- A release bump edits `__init__.py` plus the human sites: the README `Current release:` line, the `docs/ARCHITECTURE.md` heading, and `CHANGELOG.md`.
- After every bump rerun `uv pip install -e .` or the branding metadata test fails by design — the installed dist-info version freezes at install time (in a bare `pythonpath=["src"]` checkout without an install, that test skips instead).

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

Test conventions: flat `tests/` (19 files, no conftest, no shared helpers). DB isolation via `tempfile.TemporaryDirectory()`; upstream faked either by `httpx.MockTransport` handler (foreground proxy) or `AsyncMock` on `learning_chat_completion`/`retriever.search` (background). Tests construct `AppConfig()` directly, mutate fields, call `create_app(cfg)`, and wrap in `TestClient`.

Before packaging a release, verify at minimum:

- `import infinitum` works;
- the `infinitum` CLI resolves;
- runtime-only headers do not leak upstream;
- an older SQLite DB opened via an explicit path still migrates in place;
- all tests pass.

## Roadmap direction

Read `docs/ROADMAP.md` before implementing larger features. Key future work includes:

- the Phase 1 evaluation loop is implemented in `benchmarks/` (golden corpus, offline runner, precision/recall + token metrics, live replay); the remaining Phase 1 item is periodic deep consolidation, deferred to its own future update;
- hard user/project/session/agent memory scopes;
- authenticated identity from a trusted LiteLLM edge;
- organization/team memory and authoritative directives/goals;
- document/source expansion and progressive retrieval tools;
- PostgreSQL/pgvector and distributed operation;
- Responses API support;
- observability and admin/inspection tooling.

Reference docs: `docs/ARCHITECTURE.md` (mechanisms), `docs/CONFIGURATION.md` (config keys), `docs/API.md` (HTTP surface and memory-tool contract), `docs/ROADMAP.md`, and `docs/REFERENCES.md`.

The long-term goal is not merely to store more history. It is to give otherwise stateless models coherent continuity while keeping memory current, explainable, bounded, and replaceable as models improve.

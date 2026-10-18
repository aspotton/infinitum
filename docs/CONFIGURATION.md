# Configuration

Every configuration surface, grouped by area. Every key and its factory default is collected in the [reference table below](#defaults-at-a-glance); the annotated starter example lives in [`../config.example.yaml`](../config.example.yaml).

## Defaults at a glance

A working setup needs only the two `upstream:` keys below; every other setting ships with a tuned default. The full surface:

| key | default | what it does |
| --- | --- | --- |
| `upstream.base_url` | `http://localhost:4000/v1` | OpenAI-compatible upstream to forward to (the one required key in practice) |
| `upstream.api_key` | `""` (empty) | key for the upstream; unnecessary when `passthrough_authorization` forwards the inbound key |
| `upstream.passthrough_authorization` | `true` | forward the inbound `Authorization` header to the upstream |
| `memory.enabled` | `true` | master memory toggle |
| `memory.database_path` | `./infinitum.db` | SQLite database location |
| `memory.minimum_relevance_score` | `0.08` | eligibility gate: at least one genuine relevance signal |
| `memory.minimum_retrieval_score` | `0.25` | precision floor for retrieval and injection |
| `memory.inject_max_memories` | `8` | maximum memories injected into one request |
| `memory.tools_enabled` | `true` | expose the read-only drill-down tools (set `false` for upstreams without a tool parser) |
| `memory.stream_reasoning` | `live` | streaming reasoning mode (`buffered` restores pre-0.2 hold) |
| `memory.full_scan_threshold` | `500` | active-count ceiling for the byte-identical full-scan path |
| `memory.episodic_retention_days` | `30` | how long unreinforced episodic memories stay active (`0` disables retirement) |
| `embeddings.enabled` | `false` | opt-in semantic retrieval |
| `learning.enabled` | `true` | asynchronous learning worker |
| `learning.skip_subsessions` | `true` | do not learn from parent-session-marked sub-sessions |
| `learning.timeout_seconds` | `600` | background learning request timeout |
| `learning.max_tokens` | `1024` | extraction/summary generation cap |
| `learning.extra_body` | `{tool_choice: none}` | fields merged into background calls (add `chat_template_kwargs` to disable thinking) |
| `learning.skip_when_upstream_busy` | `true` | do not claim learning jobs while a request is in flight |
| `learning.upstream_idle_grace_seconds` | `5` | keep learning deferred until the upstream has been idle this long |
| `learning.topic_summaries` | `true` | incremental topic summaries |
| `learning.topic_summary_min_memories` | `3` | minimum active memories before a topic gets a summary |
| `learning.topic_summary_debounce_seconds` | `60` | quiet period before summarizing a burst of changes |
| `learning.topic_summary_update_threshold` | `3` | dirty memories that make a summary job immediately eligible |
| `learning.topic_summary_max_changed_memories` | `6` | maximum dirty records consumed by one summary call |
| `learning.topic_summary_context_memories` | `4` | active-topic sample supplied beside the changed records |
| `learning.topic_summary_bootstrap_max_memories` | `8` | bounded bootstrap sample for a summary-less topic |
| `learning.topic_summary_max_tokens` | `512` | topic-summary generation cap |
| `learning.topic_summary_fallback_memories` | `12` | active memories used when the model returns no summary text |
| `learning.consolidation` | `false` | periodic deep consolidation master gate |
| `learning.consolidation_max_tokens` | `2048` | token cap for the consolidation generation itself |

## Minimal config

```yaml
upstream:
  base_url: http://localhost:4000/v1
  api_key: ${UPSTREAM_API_KEY:-}
```

These two keys are the whole minimal setup; everything else takes the tuned defaults above. `upstream.passthrough_authorization` already defaults to `true`, so the inbound `Authorization` header is forwarded to the upstream without being listed. Background learning cannot reuse a per-request key after the request ends, so either configure a service key under `learning.api_key` or set `upstream.api_key` when the learning endpoint requires authentication.

The annotated example in [`../config.example.yaml`](../config.example.yaml) writes `database_path` as `${INFINITUM_DATABASE_PATH:-./infinitum.db}`, so container users can pin the database location via environment variable without editing the file (see [DOCKER.md](DOCKER.md)).

Storage maintenance (transparent lzma compression of event content over 2048 characters into `events.content_blob`, plus the 7-day sweep of finished background jobs) is unconditional startup behavior with no configuration keys. There is nothing to enable or tune here; see [ARCHITECTURE.md](ARCHITECTURE.md#event-content-compression) for the semantics.

## Request-context and OpenCode headers

V0.2.0 can associate an OpenAI request with a user/project/CWD while keeping the memory store globally visible. Prefer the canonical headers:

```text
X-Infinitum-User-ID: example-user
X-Infinitum-Project-ID: infinitum
X-Infinitum-CWD: /home/example/infinitum
```

An explicit project ID is preferred. If it is omitted but CWD is supplied, the runtime normalizes the path and derives a stable local key such as `cwd:infinitum:<hash>`. The full CWD is stored separately for provenance.

The default resolver also accepts common aliases so an existing OpenCode/Headroom setup can be reused:

```text
X-OpenCode-User-ID / X-OpenCode-User
X-OpenCode-Project-ID / X-OpenCode-Project
X-OpenCode-Directory / X-OpenCode-CWD

X-Headroom-User-ID
X-Headroom-Project-ID
X-Headroom-CWD

X-LiteLLM-User-ID
```

Header priority is the order configured under `request_context.*_headers`; canonical `X-Infinitum-*` values are first by default. Header names are configurable. OpenCode's own server/client path uses `x-opencode-directory` for directory context, so V0.2.0 recognizes it directly.

For session continuity, the Chat Completions route also recognizes `X-Infinitum-Session-ID`, `X-OpenCode-Session`, `X-Session-Id`, and `X-Session-Affinity` in that order. This lets OpenCode's normal compatible-provider session headers become the Infinitum `session_id` without extra client configuration.

A typical OpenCode provider configuration can send the local user and launch directory as custom headers, for example:

```jsonc
{
  "provider": {
    "infinitum": {
      "options": {
        "baseURL": "http://infinitum:8788/v1",
        "headers": {
          "x-infinitum-user-id": "{env:USER}",
          "x-infinitum-cwd": "{env:PWD}"
        }
      }
    }
  }
}
```

If you have a stable project identifier, send `x-infinitum-project-id` as well rather than depending only on CWD. `PWD` reflects the environment presented to the OpenCode process; it is not a repository-discovery protocol.

The resolved context affects retrieval only as a **soft affinity**:

```text
normal global relevance scoring
        |
        +-- below relevance floor -> reject
        |
        +-- relevant -> add bounded affinity bonus
                         same project > same user > exact CWD
```

This means a same-project memory can outrank an equally relevant memory learned elsewhere, but an unrelated same-project memory cannot become eligible solely because of project affinity. All V0.2.0 memory is still globally visible. These headers are therefore **not authentication and not a security boundary**. Hard user/project isolation is still Phase 3/4 roadmap work and must filter scope before semantic retrieval.

Useful configuration:

```yaml
request_context:
  enabled: true
  derive_project_from_cwd: true
  user_affinity_bonus: 0.03
  project_affinity_bonus: 0.07
  cwd_affinity_bonus: 0.01
  forward_to_headroom: false
```

Consumed aliases are stripped from normal upstream forwarding. When the immediate upstream is Headroom, set `forward_to_headroom: true`; Infinitum will emit canonical `x-headroom-*` values from the resolved context instead of blindly forwarding inbound identity-like headers.

`X-Infinitum-Debug: true` additionally returns the resolved user/project IDs (when present) and whether the project was derived from CWD. `GET /request-context` can be called directly to inspect resolution without invoking a model.

## Embeddings

Embeddings are optional and, in this build, supported but not yet exercised in day-to-day use — the wiring is fully in place (config, write-on-create, embed-on-search, and a semantic term in the hybrid scorer), but it ships disabled by default and the retrieval tuning so far has been done without them. Without embeddings, retrieval stays fully functional using lexical, topic, recency, importance, and confidence signals.

Semantic search is the single largest relevance signal available to Infinitum (it carries the heaviest retrieval weight), so it is intentionally left as head room rather than something already maxed out. Every accuracy and recall improvement tuned so far — the relevance gate, lexical and topic scoring, freshness decay, and the reinforcement guards — was developed and calibrated with that signal absent. Turning embeddings on and testing them against a real embedding server is expected to make recall and ranking measurably better, especially for memories that matter but are phrased differently from today's query, where lexical matching alone can miss the connection.

```yaml
embeddings:
  enabled: true
  base_url: http://embedding-server:8000/v1
  api_key: ${EMBEDDING_API_KEY:-}
  model: text-embedding-3-small
```

Any OpenAI-compatible `/v1/embeddings` implementation can be used. When enabled, semantic similarity joins the hybrid blend of lexical, topic, importance, confidence, freshness, and optional provenance-affinity signals described in the [README's retrieval section](../README.md); when a memory has no usable vector or an embedding call fails, scoring degrades gracefully to the non-semantic signals rather than dropping the memory.

## Memory drill-down tools

The read-only drill-down tool behavior contract is in [API.md](./API.md); the keys below gate it. The tools are now on by default; set `memory.tools_enabled: false` to restore the prior tool-free behavior for upstreams without a tool parser.

```yaml
memory:
  tools_enabled: true  # default: true
  # "live" streams model reasoning deltas before a round decides;
  # "buffered" holds everything until the decision (pre-0.2 behavior).
  stream_reasoning: live  # default: live
  # SSE delta field names that carry model reasoning for the live mode.
  reasoning_delta_fields: [reasoning, reasoning_content]
```

Setting `stream_reasoning: buffered` restores the pre-0.2 hold-everything behavior, where streamed rounds are withheld until the tool-round decision resolves.

## Separate learning model

```yaml
learning:
  enabled: true
  base_url: http://litellm:4000/v1
  api_key: ${LEARNING_API_KEY:-}
  model: memory-extractor
  timeout_seconds: 600
  max_tokens: 1024
```

If `learning.model` is empty, the original answering model is reused.

## Slow local learning models and timeouts

Memory extraction runs after the foreground response and uses a non-streaming
Chat Completions request. V0.1.1+ gives learning its own timeout and caps the
completion length so reasoning/local models cannot silently inherit an
unbounded generation budget.

```yaml
learning:
  timeout_seconds: 600
  max_tokens: 1024
  extra_body:
    tool_choice: none  # already the default; only chat_template_kwargs is an opt-in addition
```

If a learning request times out, the user's original chat response is unaffected.
The durable job remains eligible for retry up to `learning.max_attempts`. For a
slow local model, increase `timeout_seconds`; if the model tends to reason at
length, configure a faster dedicated extraction model or disable thinking for
background learning when your OpenAI-compatible server exposes such a control.

For OpenAI-compatible servers that support `chat_template_kwargs` (or an equivalent vendor switch), a useful setup is:

```yaml
learning:
  extra_body:
    tool_choice: none
    chat_template_kwargs:
      enable_thinking: false
```

`extra_body` is merged only into background learning Chat Completions. Infinitum's
core `model`, `messages`, `stream: false`, and configured token cap remain authoritative.

## Deferring learning while the upstream is busy

When the extraction model is the same server as the answering model, background
learning competes with foreground requests for the same GPU/CPU. With
`skip_when_upstream_busy` enabled, the learning worker skips claiming jobs
while a proxy request is in flight:

```yaml
learning:
  enabled: true
  skip_when_upstream_busy: true
```

Deferred turns are never lost: learning work lives in the durable job queue, so
a skipped job simply stays pending and runs once the proxy is idle. The boundary
is honest. Only job *start* is deferred: a job already running continues to
completion within `learning.timeout_seconds`, and a hung streaming request keeps
learning deferred while it counts as active. `GET /health` reports the live
`active_requests` count so you can see why the worker is idle. Set
`learning.upstream_idle_grace_seconds` (default `5`, pairing with
`skip_when_upstream_busy` which defaults `true`) to keep learning deferred
until the upstream has been continuously idle for that long after traffic
drains; any new foreground request restarts the window. Set it to `0` to
claim as soon as the active-request counter drains.

The mechanism internals are covered in [ARCHITECTURE.md](./ARCHITECTURE.md#deferring-learning-under-upstream-contention).

## Skipping learning for sub-sessions

Some clients spawn child agent sessions (e.g. OpenCode's task tool) that send a
parent-session marker — `X-Infinitum-Parent-Session-ID`, or the bare
`x-parent-session-id` alias — alongside the request. By default these
sub-sessions do not enqueue `learn_interaction` jobs, so delegated work does not
dilute learning with second-hand context:

```yaml
learning:
  skip_subsessions: true  # default
```

Events and retrieval are unaffected: sub-session requests are still recorded in
full and still receive injected memory. An explicit `X-Infinitum-Learning: on`
header outranks the skip; `off` outranks everything. Set
`skip_subsessions: false` to learn from sub-sessions like any other request.

### Sub-agent and harness integration

The general contract is harness-side and works with any client that supports
per-model or per-route provider options: point synthetic and sub-worker
traffic at a provider entry that sends `X-Infinitum-Learning: off`. Infinitum
additionally auto-skips requests that carry the parent-session marker when
`skip_subsessions` is true, so a harness whose sub-sessions self-report a
parent session needs no extra configuration at all. The full precedence is:
explicit `off` beats explicit `on`, which beats the sub-session skip, which
beats default learning. Unparseable header values count as absent.

OpenCode is one worked example of that contract; the same shape applies to any
client with per-model provider options:

```jsonc
{
  "provider": {
    // Learning-enabled entry for interactive sessions.
    "infinitum": {
      "npm": "@ai-sdk/openai-compatible",
      "options": { "baseURL": "http://localhost:8788/v1" },
      "models": { "your-model": {} }
    },
    // Same baseURL, learning pinned off, for synthetic traffic.
    "infinitum-no-learn": {
      "npm": "@ai-sdk/openai-compatible",
      "options": {
        "baseURL": "http://localhost:8788/v1",
        "headers": { "x-infinitum-learning": "off" }
      },
      "models": { "your-model": {} }
    }
  },
  // Route internal (title, summary, compaction) calls at the no-learn entry.
  "small_model": "infinitum-no-learn/your-model",
  "agent": {
    // An agent's own model wins over small_model, so a stale pin here
    // keeps that internal call on the learning entry. Override them all.
    "title": { "model": "infinitum-no-learn/your-model" },
    "summary": { "model": "infinitum-no-learn/your-model" },
    "compaction": { "model": "infinitum-no-learn/your-model" },
    // Delegate subagents the same way; .md agent files use the same
    // model: frontmatter key.
    "your-subagent": { "model": "infinitum-no-learn/your-model" }
  }
}
```

Note the version caveat: OpenCode's automatic parent-session marker ships in
releases after 2026-08. Older builds send sub-sessions without the marker, so
the no-learn provider entries above are what keep those harnesses quiet.

Verify this on your build rather than trusting the config alone: trigger a
compaction, then confirm the synthetic summary prompt shows up in
`GET /events` as a `message.user` event with **no** matching
`learn_interaction` job. Whether a given build's compaction call sites honor
the model pins is build-dependent, so check the events, not the config. As a
generic fallback, a harness can simply send the `X-Infinitum-Learning: off`
header itself on synthetic calls; some OpenCode builds also expose an
undocumented plugin hook keyed on agent name that can attach headers, if your
version has it.

Any OpenAI SDK client can do the same in two lines:

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8788/v1", default_headers={"X-Infinitum-Learning": "off"})
```

## Incremental topic-summary controls

V0.1.2+ no longer regenerates a topic summary from up to 100 topic memories after every learned interaction. Changed memory IDs are persisted as dirty topic state, and one debounced background job updates the existing summary.

```yaml
learning:
  topic_summaries: true
  topic_summary_min_memories: 3
  topic_summary_debounce_seconds: 60
  topic_summary_update_threshold: 3
  topic_summary_max_changed_memories: 6
  topic_summary_context_memories: 4
  topic_summary_bootstrap_max_memories: 8
  topic_summary_max_tokens: 512
  topic_summary_fallback_memories: 12
```

- `debounce_seconds`: wait for a quiet period before summarizing a burst of changes.
- `update_threshold`: if this many dirty memories accumulate, make the summary job immediately eligible.
- `max_changed_memories`: maximum dirty records consumed by one summary call.
- `context_memories`: small active-topic sample supplied beside the changed records.
- `bootstrap_max_memories`: bounded sample used only when a topic has no existing summary yet.
- `fallback_memories`: maximum active canonical memories used when the learning model returns no final summary text.

Dirty state is cleared only after a usable model summary or the deterministic active-memory fallback has been persisted. If new evidence arrives while a summary is running, it remains dirty and is scheduled for a follow-up rather than being lost.

## Periodic consolidation controls

Periodic deep consolidation is the third timescale: a slow sweep over stale, changed topics that clusters near-duplicates, canonicalizes them, and records conflicts. It is opt-in and requires `learning.enabled`, because the learner worker hosts the sweep.

```yaml
learning:
  consolidation: true
  consolidation_interval_seconds: 604800
  consolidation_min_changed_memories: 8
  consolidation_max_memories_per_pass: 60
  consolidation_max_tokens: 2048
```

- `consolidation`: master gate, `false` by default. When unset, behavior is byte-identical to pre-consolidation builds. Enabling it also requires `learning.enabled` and an explicit `learning.model`: unlike per-interaction learning, a background sweep has no request model to reuse, so with the model unset the sweep stays silently idle.
- `consolidation_interval_seconds`: how often one topic is eligible for a pass (default 7 days). The sweep enqueues at most one topic per minute, never while `learning.model` is unset, and only for topics at or above the churn floor. Sub-floor topics are never enqueued at all, so their pending changes accumulate instead of being discarded.
- `consolidation_min_changed_memories`: churn floor. A topic needs at least this many changed memories since its last checkpoint to be worth a pass, checked at enqueue time and again when the pass runs. The checkpoint advances only on executed passes: a churn-floor no-op writes its audit event without stamping, so sub-floor churn keeps accumulating toward the next real pass, while an executed pass stamps even when it finds zero clusters, so floored work is never re-proposed every tick. Consolidation's own writes do **not** count as churn for the next window because the checkpoint is stamped after mutations, so oversized topics advance via explicit continuation jobs that bypass only the churn floor.
- `consolidation_max_memories_per_pass`: bounded working set consumed by one pass.
- `consolidation_max_tokens`: token cap for the consolidation generation itself.

Conflicts found during consolidation are recorded as `consolidation.conflict` events and are otherwise inert; nothing is auto-resolved.

## Reinforcement tuning knobs

Useful tuning controls:

```yaml
memory:
  reinforce_similarity: 0.86
  reinforce_semantic_similarity: 0.90
  reinforce_hint_min_score: 0.55
  reinforce_hint_min_lexical: 0.40
  reinforce_hint_min_semantic: 0.72
```

The `memory.supersede_similarity_floor` knob (default `0.30`), which already gates implicit supersede similarity, also gates refine routing: an unmarked supersede proposal naming a single shown, active, same type/topic target refines that memory in place only when lexical similarity sits at or above this floor (issue #34).

## Topic canonicalization knobs

Topic slugs that differ only by word order, separator, or affix variants are the same topic. Without canonicalization, exact-string topic gates never match slug variants, which is why 87% of active memories in a measured live store have `observation_count = 1` — equivalent facts keep landing in sibling topics instead of reinforcing one memory.

```yaml
memory:
  topic_canonical_floor: 0.85
  topic_canonicalization_prompt_topics: 40
```

- `memory.topic_canonical_floor` (default `0.85`) — slug-aware similarity required to treat two topics as equivalent when merging at apply time and during the one-time startup backfill. This is deliberately separate from `supersede_similarity_floor`; topic slugs are short strings where a 0.30 floor would collapse unrelated topics.
- `memory.topic_canonicalization_prompt_topics` (default `40`) — how many existing topics are listed in the extraction prompt so the extractor reuses canonical topic names instead of inventing near-duplicates. Set to `0` to disable the block.

## Episodic retention

```yaml
memory:
  episodic_retention_days: 30  # default; 0 disables retirement
```

Episodic memories are session-continuity records, not durable facts, so the runtime retires the ones that stopped being useful. Retirement is a deterministic pass on the learning worker's tick (every 60 seconds, up to 200 rows per tick, no LLM involvement): an **active** `episodic` memory is archived when it has never been reinforced (`observation_count < 2`) **and** both its `updated_at` and its last access (or its creation, for rows never injected) fall outside the `episodic_retention_days` window. Archiving is a status flip to `archived`: the memory stops being injected or retrieved, but the raw events that produced it are untouched and it stays inspectable via `GET /memory?status=archived`.

What is exempt from retirement:

- any memory reinforced at least once (`observation_count >= 2`), episodic or not: evidence of actual reuse;
- anything created, updated, or injected within the window: a never-reinforced episodic that keeps getting retrieved stays active;
- every non-episodic memory type: retirement touches `episodic` rows only.

The same window drives the learner-side duplicate-skip guard: an episodic candidate that is near-verbatim (`memory.dedup_similarity`) to a still-fresh active episodic is skipped at apply time instead of minting a duplicate, and the diagnostic counter is surfaced as `skips.episodic_total` on `GET /status`. Set `episodic_retention_days: 0` to disable both the retirement pass and the guard (the guard's cutoff becomes `now`, so nothing is ever fresh enough to match). The window is deliberately shared and the two roles are complements: inside the window an episodic is a live anchor that absorbs near-duplicates; outside it, an unreinforced episodic is retirement fodder. Mechanics: [ARCHITECTURE.md](ARCHITECTURE.md#episodic-lifecycle).

## Retrieval candidate fast path

```yaml
memory:
  full_scan_threshold: 500
```

- `memory.full_scan_threshold` (default `500`) — stores with an active-memory count at or below this value keep the byte-identical full-scan retrieval path; above it, retrieval scores a provably-complete candidate shortlist instead of every active memory (mechanics in [ARCHITECTURE.md](ARCHITECTURE.md#relevance-eligibility)). No configuration is required to benefit. Setting a huge value (e.g. `1000000000`) is the kill switch: every store takes the full-scan path. The shortlist also yields to the full scan whenever `memory.minimum_relevance_score` sits below `0.08`, because the shortlist's completeness proof rests on that floor.

## Sample setup that works for me

The primary maintainer runs a local Qwen model on an NVIDIA DGX Spark, with Infinitum pointing directly at it. This is the working configuration for that single-machine setup:

```yaml
upstream:
  base_url: http://127.0.0.1:8889/v1
  passthrough_authorization: true

memory:
  database_path: /home/example/infinitum.db
  minimum_retrieval_score: 0.30
  minimum_relevance_score: 0.08
  inject_max_memories: 6
  tools_enabled: true

learning:
  enabled: true
  timeout_seconds: 600
  max_tokens: 1024
  skip_when_upstream_busy: true
  upstream_idle_grace_seconds: 5

  topic_summaries: true
  topic_summary_min_memories: 3

  topic_summary_debounce_seconds: 30
  topic_summary_update_threshold: 3

  topic_summary_max_changed_memories: 6
  topic_summary_context_memories: 4

  topic_summary_bootstrap_max_memories: 8
  topic_summary_max_tokens: 512

  extra_body:
    tool_choice: none
    chat_template_kwargs:
      enable_thinking: false
```

Your paths, ports, and tuning will differ; every key is documented in the sections above and in [`../config.example.yaml`](../config.example.yaml).

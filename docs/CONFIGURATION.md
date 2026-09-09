# Configuration

Every configuration surface, grouped by area. The annotated key-by-key example lives in [`../config.example.yaml`](../config.example.yaml).

## Minimal config

```yaml
upstream:
  base_url: http://localhost:4000/v1
  passthrough_authorization: true

memory:
  database_path: ./infinitum.db

learning:
  enabled: true

embeddings:
  enabled: false
```

With `passthrough_authorization: true`, the inbound `Authorization` header is forwarded to the upstream. Background learning cannot reuse a per-request key after the request ends, so either configure a service key under `learning.api_key` or configure `upstream.api_key` when the learning endpoint requires authentication.

## Request-context and OpenCode headers

V0.2.0 can associate an OpenAI request with a user/project/CWD while keeping the memory store globally visible. Prefer the canonical headers:

```text
X-Infinitum-User-ID: adam
X-Infinitum-Project-ID: infinitum
X-Infinitum-CWD: /home/adam/infinitum
```

Pre-v0.2 `X-Context-User-ID`, `X-Context-Project-ID`, and `X-Context-CWD` are still accepted as lower-priority compatibility aliases.

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

## Separate learning model

```yaml
learning:
  enabled: true
  base_url: http://litellm:4000/v1
  api_key: ${LEARNING_API_KEY:-}
  model: memory-extractor
  timeout_seconds: 600
  max_tokens: 2048
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
  max_tokens: 2048
  extra_body: {}
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
`learning.upstream_idle_grace_seconds` (default `0`) to keep learning deferred
until the upstream has been continuously idle for that long after traffic
drains; any new foreground request restarts the window.

The mechanism internals are covered in [ARCHITECTURE.md](./ARCHITECTURE.md#deferring-learning-under-upstream-contention).

## Incremental topic-summary controls

V0.1.2+ no longer regenerates a topic summary from up to 100 topic memories after every learned interaction. Changed memory IDs are persisted as dirty topic state, and one debounced background job updates the existing summary.

```yaml
learning:
  topic_summaries: true
  topic_summary_min_memories: 3
  topic_summary_debounce_seconds: 30
  topic_summary_update_threshold: 5
  topic_summary_max_changed_memories: 24
  topic_summary_context_memories: 8
  topic_summary_bootstrap_max_memories: 32
  topic_summary_max_tokens: 1024
  topic_summary_fallback_memories: 12
```

- `debounce_seconds`: wait for a quiet period before summarizing a burst of changes.
- `update_threshold`: if this many dirty memories accumulate, make the summary job immediately eligible.
- `max_changed_memories`: maximum dirty records consumed by one summary call.
- `context_memories`: small active-topic sample supplied beside the changed records.
- `bootstrap_max_memories`: bounded sample used only when a topic has no existing summary yet.
- `fallback_memories`: maximum active canonical memories used when the learning model returns no final summary text.

Dirty state is cleared only after a usable model summary or the deterministic active-memory fallback has been persisted. If new evidence arrives while a summary is running, it remains dirty and is scheduled for a follow-up rather than being lost.

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

## Sample setup that works for me

The primary maintainer runs a local Qwen model on an NVIDIA DGX Spark, with Infinitum pointing directly at it. This is the working configuration for that single-machine setup:

```yaml
upstream:
  base_url: http://127.0.0.1:8889/v1
  passthrough_authorization: true

memory:
  database_path: /home/adam/infinitum.db
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

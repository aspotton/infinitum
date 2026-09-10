# benchmarks

Dev-only evaluation harness for the memory pipeline. This directory is **not an
installed package**: every command must run from the repository root, where the
repo root lands on `sys.path`. Use the project virtualenv explicitly:

## Offline eval (deterministic, no network)

```bash
.venv/bin/python -m benchmarks.run                       # all scenarios
.venv/bin/python -m benchmarks.run --scenario durable-fact   # repeatable
.venv/bin/python -m benchmarks.run --json report.json    # also dump report JSON
.venv/bin/python -m benchmarks.run --scenarios-dir some/dir  # alternate corpus
```

Each scenario runs against an in-process instance with a scripted extractor, so
results are deterministic and require no upstream model. The report prints
per-scenario pass/fail records plus aggregate extraction/retrieval
precision-recall and total context tokens. Exit code: `0` all green, `1` if any
scenario failed, `2` for a bad scenario name or a scenario file that fails to
load. Every scenario file in `benchmarks/scenarios/` is also gated inside the
normal test suite (`tests/test_eval_corpus.py`), so a regression shows up as a
scenario-named pytest failure without anyone remembering to run the CLI.

## Live replay (against a running instance)

```bash
.venv/bin/python -m benchmarks.replay --base-url http://127.0.0.1:8788
.venv/bin/python -m benchmarks.replay --scenario explicit-correction --strict
```

Replay posts each scenario turn to `/v1/chat/completions` under
`X-Infinitum-User-ID: eval` and `X-Infinitum-Project-ID: <scenario name>`, then
prints the memories newly created or changed after every turn. The diff
baseline starts **empty per scenario**, so turn 1 shows whatever is already in
the store as new rows; later turns print `(no memory changes)` unless the live
learner actually wrote memory. Re-running is safe (read-only beyond the chat
turns themselves).

Replay is a review aid, not a verdict: expectation mismatches print `WARN`
lines and replay exits `0` anyway, because live learning is non-deterministic
and depends on the configured extraction model. Pass `--strict` to make any
`WARN` exit non-zero. A scenario whose server is unreachable prints an `ERROR`
line, is skipped, and does not affect the exit code; the remaining scenarios
are still attempted.

## Scenario corpus format

Scenarios are YAML files in `benchmarks/scenarios/`. The authoritative field
list is the Pydantic model set in [`corpus.py`](corpus.py): a scenario is a
`name`, `description`, `context` (user/project ids), and one or more `turns`,
each with `user`/`assistant` text, optional `learn` candidate specs (what the
offline scripted extractor proposes), `expect` outcome checks (`created`,
`reinforced`, `superseded`, `ignored`), and an optional retrieval `probe`
(`must_include`/`must_not_include`, `temporal_view`, `as_of`). Unknown fields
are rejected at load time, and dates (`valid_from`, `valid_until`, `as_of`)
must be quoted ISO `YYYY-MM-DD` strings.

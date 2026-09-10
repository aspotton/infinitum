---
name: infinitum-eval
description: Evaluate a live Infinitum memory store (SQLite DB or running instance) against the benchmarks/scenarios expectations with semantic equivalence instead of exact string matching. Use when asked to evaluate/grade/audit live memory quality or benchmark equivalence against a real model's learned memories.
---

# Infinitum live-store evaluation

## Purpose

`python -m benchmarks.replay` compares live memory content to the corpus
expectations with byte-exact substring checks, so a real model that learned the
right fact in different words prints a `WARN`. Those WARNs are artifacts of the
matcher, not verdicts about learning. This skill grades what the live store
actually learned against what each scenario expects, using your own reading and
judgment instead of string equality. The machine verdict stays
`python -m benchmarks.run` (the deterministic offline gate, also wired into
`tests/test_eval_corpus.py`); this is the human-facing semantic audit.

## Read-only pledge

Never run INSERT, UPDATE, or DELETE against the database. Never POST to any
endpoint except `POST /memory/search` probes. Never edit scenario YAMLs. If a
store looks wrong, report it; do not fix it.

## Locate the store

Default: `./infinitum.db` in the repository root. If the `INFINITUM_CONFIG`
environment variable is set, read `memory.database_path` from that YAML/JSON
config file and use it instead.

First check size:

```bash
sqlite3 -readonly <db> "SELECT COUNT(*) FROM memories;"
```

If the memories table is empty or has fewer than 3 rows, stop. Tell the user the
store is not seeded and give them this: seed first with
`.venv/bin/python -m benchmarks.replay --base-url <url> --model <model> --timeout 120`,
wait for the learning worker to go idle, then re-run this skill. Do not grade an
unseeded store; the result would be a wall of false FAILs.

## Read expectations

Read every file in `benchmarks/scenarios/*.yaml` (10 files). The schema is
defined in `benchmarks/corpus.py`: each scenario has `name` (also its
`context.project_id`), `description`, `context`, and one or more `turns`. Each
turn has `user` text carrying the candidate facts, an `expect` block with
`created` / `superseded` / `reinforced` lists of expected fact strings (plus
`ignored`), and turns may carry a `probe` with `query`, `as_of`,
`must_include`, and `must_not_include`. Quote expected strings verbatim from
the YAML. Do not invent expectations.

## Read actual memories

Dump the store:

```bash
sqlite3 -readonly <db> "SELECT id, status, memory_type, topic, content, observation_count FROM memories ORDER BY status, updated_at DESC;"
```

The `-readonly` flag is mandatory. Use `sqlite3 -readonly <db> "SELECT
memory_id, COUNT(*) FROM memory_observations GROUP BY memory_id;"` when you
need to count evidence rows behind a memory.

For probe expectations against a running server, use `POST /memory/search` with
headers `X-Infinitum-User-ID: eval` and `X-Infinitum-Project-ID: <scenario.name>`
and body `{"query": "<probe.query>"}`, adding `temporal_view` and `as_of` when
the probe sets them. For DB-only grading with no server, judge the probe query
lexically against the memory list you dumped and label the result
`probe(db-judged)`.

## Grading rubric

Go scenario by scenario, expectation by expectation:

- `created`: at least one **active** memory conveys the same fact.
- `superseded`: at least one memory row with `status != 'active'` carries the
  old fact AND an active memory carries the replacement fact.
- `reinforced`: the memory covering the fact has `observation_count >= 2`.
  Legacy-backfill rows can show count 1 while carrying weight-0.5 legacy
  observations, so if the content matches but the count is 1, check
  `memory_observations` for more than one row before failing.
- `probe`: the search (or db-judged fallback) returns a memory conveying the
  expected fact. Honor `as_of` by preferring rows whose `valid_from` /
  `valid_until` window contains it when those columns are present.

**Precision rule** (apply to every grade above): wording may be rephrased
freely; every number, version, region code, date, and proper noun in the
expectation must appear in the matched memory in a fact-equivalent form ("runs
on PostgreSQL 16" ≡ "PostgreSQL 16 powers it"; "PostgreSQL 14" is FAIL, not
paraphrase). A paraphrase that drops or alters one of these tokens is FAIL.

Scenario verdict per expectation:

- PASS: byte or near-byte match.
- PASS(paraphrased): semantic match with the precision rule held.
- FAIL: missing fact, wrong fact, or wrong status.

## Output format

Report one markdown table:

```
| scenario | expectation | verdict | memory id(s) | why (one line) |
```

Close with a totals line in the form `X PASS / Y paraphrased / Z FAIL of N`.
Do not write any files unless the user asks.

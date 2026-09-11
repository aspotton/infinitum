# Issues — asymmetric-temporal-demotion

Problems and gotchas encountered during work on this plan.

_Auto-scaffolded by /start-work. Append new entries below - never overwrite._

---

## 2026-09-11 Task: todo1 (verified, marked)
- EXPECTED intermediate failure: tests/test_eval_corpus.py::test_scenario_gate[temporal-tool-derived] (must_not_include on expired fact now demoted-present). Resolved by todos 3+4. Do NOT count as regression in todo 6 sweeps before todo 4 lands. temporal-validity scenario currently still passes its gate.
- LSP daemon unresponsive (timeouts); pytest full-suite is the effective type/import gate. ruff NOT installed in .venv (todo 6(c): record not-installed baseline).
> [todo6] Full sweep green at 0f9c491, zero fixes: pytest 330/330, benchmarks 11/11, diff=exactly-18-allowlist files, protected files empty, determinism+0.70-ratio spot check PASS (fixed-clock needed for byte-identical — wall-clock freshness drift breaks round(score,12) across calls otherwise), mutation QA at compiler.py:161 fails exactly the named test then restores clean. Temp scripts in /tmp/opencode only, deleted.

---

## 2026-09-11 Final wave F1-F4
- All four reviewers APPROVE (parallel background). F3 live-route QA: expired ratio 0.70000000 exact, as_of undemoted, chat block rank1-active/rank2-expired-present.
- Non-blocking notes: F2 nits (E501-consistent long line compiler.py:161, one 0.7-vs-0.70 literal, tense); F3 finding: POST /memory (MemoryCreateRequest) cannot set valid_from/valid_until — pre-existing API-surface gap, seed via Database in tests/scripts; F1 note: demotion_violations matches by first content-containment (dedup makes id-collision unreachable).
- Plan F1-F4 set to [~] = reviews complete, awaiting the user okay gate mandated by the plan header before [x].

## 2026-09-11 Task 8: successor-ordering follow-up (items A-D)
- must_rank_below is position-based in the PROBED view only — one search POST, no
  all-view follow-up; asserting "exactly one POST without temporal_view" locks that
  machine contract (test_replay). Inverted-expectation twin on identical data is the
  cleanest provably-failing case for a checker (no data surgery needed).
- demotion_violations now returns dict[entry, reason]; membership checks (`sub in
  dict`) are call-compatible with the old set, so the ceiling guard cost zero caller
  churn beyond dropping set().
- Trade-off pin arithmetic: with a fixed wording pair, the imp/conf delta (0.14+0.10
  weight x 0.4 drops = 0.096) IS the whole window — both margins must land under
  0.096, so the realistic margin and the flip margin trade against each other.
  Winning pair gave +0.0655/+0.0305; the first candidate pair missed by 0.002.
- Pin lesson: if a pin also asserts current == all * 0.70 with a literal, it fails
  under any factor mutation (even "stronger factor = still wins" spot-checks) — use
  retrieval_mod.EXPIRED_FACTOR in the ratio check when ORDER, not the constant, is
  the pin.
- One-off test failure immediately after an edit+sed chain (5x reruns + full-file
  green) — treat first-run-after-save failures as suspect until reproduced; sed
  mutation probes restore byte-clean (git diff src empty).

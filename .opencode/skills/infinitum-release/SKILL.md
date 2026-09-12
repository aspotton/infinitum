---
name: infinitum-release
description: Cut a full Infinitum release from a base branch — version-bump PR, merge, GitHub release publish, then monitor the GHCR build workflow to green. Use when asked to release, ship, bump-and-publish, or cut a new version of Infinitum.
---

# Infinitum release

## Purpose

Publish a new Infinitum version end to end: prepare a version-bump PR (the repo
never receives direct pushes), merge it, publish the GitHub release, and watch
the `.github/workflows/release.yml` build to green. Publishing the release is
what triggers the multi-arch GHCR image build — nothing ships until that run is
green. This skill does **not** pull or smoke-test the image locally; the CI run
is the gate.

## Inputs

- Base branch: default `main`.
- Version: explicit if the user gives one. Otherwise derive it: read the
  `## Unreleased` section of `CHANGELOG.md` and the current `__version__`.
  `Added:` items present → minor bump; only `Changed:`/`Fixed:` → patch bump.
  Example: `0.3.0` + features → `0.4.0`.

Use `X.Y.Z` below for the resolved version without a `v` prefix; tags and image
tags carry the `v`.

## Preflight — abort and report if any check fails

```bash
git status --short                      # must be empty
gh auth status                          # must be authenticated
git switch <base> && git pull --ff-only # on base branch at origin tip
```

Check `CHANGELOG.md` has a non-empty `## Unreleased` section — if not, there is
nothing to release; stop and say so. Remember the AGENTS.md versioning rule:
the single source of truth is `__version__` in `src/infinitum/__init__.py`;
never hardcode versions in routes or app construction.

## Step 1 — release prep PR

Branch from the base:

```bash
git switch -c release/vX.Y.Z
```

Edit **exactly** these four files — nothing else:

1. `src/infinitum/__init__.py` — set `__version__ = "X.Y.Z"`.
2. `CHANGELOG.md` — rename the `## Unreleased` heading to `## X.Y.Z`. Bodies
   stay verbatim. Do **not** add a new empty Unreleased section; do not touch
   older sections.
3. `README.md` — the `Current release: **vX.Y.Z**.` line.
4. `docs/ARCHITECTURE.md` — the line-1 heading `# Infinitum vX.Y.Z Architecture`.

**Note:** `.github/workflows/release.yml` is **not** an edit target — its
`v{{version}}` pattern already publishes `vX.Y.Z` tags matching the docs, and
the Constraints section forbids editing `.github/workflows/` during a release.

Do **not** edit `docs/DOCKER.md` — it documents `latest` as the default tag
plus the full-semver and `sha-<commit>` pin formats, so it needs no per-release
change.

Re-freeze the installed dist-info (the branding metadata test reads it and
fails against a stale version otherwise), then verify and open the PR:

```bash
uv pip install -e .
.venv/bin/python -m pytest -q           # all green required
git diff --stat                         # must be exactly those files (4 changed)
git commit -am "chore(release): vX.Y.Z"
git push -u origin release/vX.Y.Z
```

If the push is rejected with an "OAuth App ... without workflow scope" error
(happens only when a branch touches `.github/workflows/`; a clean release
branch normally pushes fine over HTTPS), push over SSH instead:

```bash
git push git@github.com:<owner>/<repo>.git release/vX.Y.Z
git fetch && git branch --set-upstream-to=origin/release/vX.Y.Z
```

```bash
gh pr create  # body: version bump, changelog summary, docs version pinning
gh pr view --json state                 # must be OPEN
```

## Step 2 — merge and publish

```bash
gh pr merge release/vX.Y.Z --merge --delete-branch
```

Merge-commit style per repo history — never squash, never rebase. If the
`--delete-branch` flag leaves the local branch, remove it with
`git branch -d release/vX.Y.Z`. Then sync and verify the installed version:

```bash
git checkout <base> && git pull --ff-only
uv pip install -e .
.venv/bin/python -c "import infinitum; print(infinitum.__version__)"  # == X.Y.Z
```

Publish the release (must be PUBLISHED, never a draft — the workflow triggers
on `release:published`):

```bash
gh release create vX.Y.Z \
  --title "vX.Y.Z — <short em-dash summary distilled from the changelog>" \
  --notes "<dash bullets distilled from CHANGELOG ## X.Y.Z, vendor-neutral>"
```

## Step 3 — monitor the build

```bash
gh run list --workflow=release.yml --limit 1   # find the run for tag vX.Y.Z
gh run watch <run-id> --exit-status --interval 20
```

Verify the run's tag matches `vX.Y.Z` before watching — an old run watching
"green" proves nothing. On success:

```bash
gh run view <run-id> --json jobs    # both `test` and `images` jobs: success
```

Report the PR URL, merge SHA, release URL, and run URL. Stop.

**On failure**: do **not** retry, re-run, revert, edit files, or delete the
release. Collect and present:

```bash
gh run view <run-id> --json jobs          # failed job/step names
gh run view <run-id> --log-failed         # trim to the failing error lines
```

Add annotations via `gh run view <run-id> --json jobs` if the log is unclear.
Present the failing job/step, the trimmed error, and the release/PR URLs — then
STOP and await instructions. A published release with a failed build is the
user's decision, not the agent's.

## Constraints

- No direct pushes to the base branch; everything goes through a PR.
- No force-push, no rebase, no amend, no squash-merge.
- No tag deletion; never publish a draft release.
- Never edit `.github/workflows/` during a release.
- All local Python runs via `.venv/bin/python` from the repo root — bare
  `python` may be absent and the system python lacks the package — always use
  the project venv.

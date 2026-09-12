# Running Infinitum with Docker

Everything Infinitum persists lives in ordinary files you own on the host: one
config file and one database directory. The container is disposable — remove it,
re-run it, and the memory store is exactly where you left it.

## 1. Image

Release images are published automatically when a GitHub release is published:

- `ghcr.io/aspotton/infinitum:vX.Y.Z` (the release tag), plus `latest` and `sha-<commit>`
- Architectures: `linux/amd64` and `linux/arm64`

## 2. Host layout

Two operator-owned paths under a directory you choose (the examples below use
`~/infinitum/`):

- `~/infinitum/config.yaml` — the config file, mounted read-only
- `~/infinitum/db/` — the SQLite database and its WAL files live here; create the empty directory as step 1

Nothing is persisted inside the image or the running container.

## 3. Initial config setup

Create the host directory, then **author** the minimal block below in
`~/infinitum/config.yaml`:

```yaml
server:
  host: 0.0.0.0
  port: 8788
upstream:
  base_url: http://host.docker.internal:4000/v1   # or your LiteLLM/vLLM endpoint
  api_key: ${UPSTREAM_API_KEY:-}
  passthrough_authorization: true
memory:
  database_path: ${INFINITUM_DATABASE_PATH:-/db/infinitum.db}
learning:
  enabled: true
embeddings:
  enabled: false
```

Do **not** copy `config.example.yaml` — its database default is the bare-host
`./infinitum.db`, which inside a container would land in the image working
directory, not in your mounted `db/` directory.

Notes on the block:

- `${...}` values are interpolated from the container environment at startup, so the file stays plaintext while keys stay env vars.
- `INFINITUM_DATABASE_PATH` is optional; unset, it defaults to the shown `/db/infinitum.db`.
- `database_path` is not tilde-expanded, so use an absolute path.
- Never put secrets in the file.

## 4. Run

The canonical command (copy-paste-ready after replacing `replace-me` with your
upstream key; works on a stock Linux host):

```bash
docker run -d --name infinitum --restart unless-stopped \
  --user $(id -u):$(id -g) \
  -p 8788:8788 \
  --add-host=host.docker.internal:host-gateway \
  -v ~/infinitum/config.yaml:/config/config.yaml:ro \
  -v ~/infinitum/db:/db \
  -e UPSTREAM_API_KEY=replace-me \
  ghcr.io/aspotton/infinitum:vX.Y.Z
```

Why `--user $(id -u):$(id -g)`: the image's built-in uid (10001) would not be
able to write a host directory you own; running the container as your own uid
is the one-flag fix. The config mount is `:ro` and world-readable by your own
uid, and the WAL files need a Linux bind mount, which this is.

Port note: the image healthcheck probes port 8788. If you set `server.port` to
something else, add `--no-healthcheck` — the service itself is unaffected; only
the `docker ps` status would lie.

## 5. Verify

Wait for health (bounded to ~30s):

```bash
timeout 30 sh -c 'until curl -sf localhost:8788/health >/dev/null; do sleep 0.5; done'
```

Then prove persistence across containers:

```bash
curl -s -X POST localhost:8788/memory -H 'content-type: application/json' \
  -d '{"content":"docker persistence check"}'
```

Note the `id` in the returned JSON, remove the container, and re-run the
canonical command from §4:

```bash
docker rm -f infinitum
```

```bash
curl -s "localhost:8788/memory/<id>"
```

The memory comes back — it was never inside the container.

## 6. If the container exits immediately

Check `docker logs infinitum` for one of these two clean one-line messages:

- `infinitum: config file not found: /config/config.yaml` — the config file bind mount is missing or at the wrong path.
- `infinitum: database directory missing or not writable: /db` — the `-v .../db:/db` mount is missing, or the host directory is owned by someone else (rerun with the `--user` flag).

Any other traceback means the config file itself is broken (e.g. YAML syntax
errors) — fix it per the message; the runtime does not paraphrase those.

## 7. Upgrade / backup / rollback

- **Upgrade**: `docker rm -f infinitum`, then re-run the canonical command with the new tag (same mounts). `--restart unless-stopped` also brings the container back after host reboots; the `docker rm -f` before an upgrade supersedes that.
- **Backup**: stop the container, then cold-copy the database files: `cp ~/infinitum/db/infinitum.db* /backup/` (the glob includes the WAL files).
- **Rollback**: re-run the previous tag. Database migrations are additive-only, so an older image reads a newer database unchanged.

## 8. Platform caveat

On macOS/Windows Docker Desktop, bind-mounted SQLite locking is unreliable —
keep `~/infinitum/db` inside the Docker VM filesystem (e.g. under the Docker
Desktop WSL/VM home), not on a plain host path.

# Migrating Context Runtime v0.1.x to Infinitum v0.2.1

Infinitum v0.2.x is a project/package rename with compatibility support. The memory/event schema remains compatible with v0.1.x.

## Recommended upgrade

1. Stop the old service cleanly.
2. Install or unpack Infinitum v0.2.1.
3. Reuse your existing `config.yaml`.
4. Keep `memory.database_path` pointed at your existing SQLite database.
5. Start with `infinitum serve --config config.yaml`.
6. Verify `/health`, `/request-context`, and a normal `/v1/chat/completions` request.
7. After validation, update clients to the canonical Infinitum names at your convenience.

No memory rewrite is required.

## Name changes

| Pre-v0.2 | Infinitum v0.2+ |
| --- | --- |
| repository/product: Context Runtime | Infinitum |
| Python package: `context_runtime` | `infinitum` |
| CLI: `context-runtime` | `infinitum` |
| config env: `CONTEXT_RUNTIME_CONFIG` | `INFINITUM_CONFIG` |
| `X-Context-*` headers | `X-Infinitum-*` headers |
| `<runtime_memory>` envelope | `<infinitum_memory>` envelope |
| default new DB: `context-runtime.db` | `infinitum.db` |

## Compatibility retained

During the 0.2 compatibility period:

- `import context_runtime` remains available as a wrapper around `infinitum`;
- the `context-runtime` CLI entry point remains available;
- `CONTEXT_RUNTIME_CONFIG` is used when `INFINITUM_CONFIG` is not set;
- `X-Context-*` request headers remain accepted as lower-priority aliases;
- existing v0.1.x SQLite databases open normally.

New code should use only the Infinitum names.

## Database filename behavior

An explicit `memory.database_path` always wins.

For an unconfigured/default startup, Infinitum normally uses `./infinitum.db`. To prevent a rename upgrade from silently appearing to lose memory, if `./context-runtime.db` exists and `./infinitum.db` does not, Infinitum automatically reuses the legacy file.

If both files exist, Infinitum uses `./infinitum.db` unless configuration explicitly selects the legacy file. This avoids guessing which database is authoritative.

## Header migration

Preferred new request context:

```text
X-Infinitum-User-ID: adam
X-Infinitum-Project-ID: my-project
X-Infinitum-CWD: /home/adam/my-project
X-Infinitum-Session-ID: session-123
```

Preferred controls:

```text
X-Infinitum-Memory: off
X-Infinitum-Learning: off
X-Infinitum-Debug: true
```

The matching `X-Context-*` forms remain accepted, but if both old and new identity headers are present the canonical `X-Infinitum-*` value takes precedence.

## OpenCode example

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

If a stable project identifier is available, add `x-infinitum-project-id` rather than relying only on CWD derivation.

## Rollback

Because the database schema remains compatible, rolling back to v0.1.4 is possible as long as the same database path is used. V0.1.4 will ignore the newer branding/API aliases but can continue using the event and memory records created by v0.2.x.

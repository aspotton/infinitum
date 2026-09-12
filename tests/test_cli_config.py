"""CLI exit-path tests for ``infinitum.__main__.main``.

Every case drives ``main()`` with an explicit ``sys.argv`` because pytest's
own argv would trip argparse, and the ``config`` attribute only exists under
the ``serve`` subparser. ``INFINITUM_CONFIG`` is always seeded (set or
deleted) so ``main()``'s env write-back is torn down by monkeypatch.
"""

from __future__ import annotations

import sys

import pytest

from infinitum.__main__ import main


def _patch_uvicorn(monkeypatch) -> dict:
    """Replace ``uvicorn.run`` in the ``__main__`` namespace, capturing kwargs.

    Prevents any real socket bind and lets us assert the guard (when present)
    fires *before* the server would start.
    """
    captured: dict = {}

    def _fake_run(*args, **kwargs):
        captured.update(kwargs)

    monkeypatch.setattr("infinitum.__main__.uvicorn.run", _fake_run)
    return captured


def test_missing_config_via_env_exits_cleanly(monkeypatch, tmp_path):
    missing = tmp_path / "nope.yaml"
    monkeypatch.setenv("INFINITUM_CONFIG", str(missing))
    monkeypatch.setattr(sys, "argv", ["infinitum", "serve"])
    _patch_uvicorn(monkeypatch)

    with pytest.raises(SystemExit) as exc:
        main()

    message = str(exc.value)
    assert "config file not found" in message
    assert str(missing) in message


def test_missing_config_via_flag_exits_cleanly(monkeypatch):
    monkeypatch.delenv("INFINITUM_CONFIG", raising=False)
    monkeypatch.setattr(sys, "argv", ["infinitum", "serve", "--config", "/missing.yaml"])
    _patch_uvicorn(monkeypatch)

    with pytest.raises(SystemExit) as exc:
        main()

    message = str(exc.value)
    assert "config file not found" in message
    assert "/missing.yaml" in message


def test_valid_config_reaches_uvicorn_with_port(monkeypatch, tmp_path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text("server:\n  port: 9\n")
    monkeypatch.delenv("INFINITUM_CONFIG", raising=False)
    monkeypatch.setattr(sys, "argv", ["infinitum", "serve", "--config", str(cfg)])
    captured = _patch_uvicorn(monkeypatch)

    main()  # must not raise SystemExit

    assert captured.get("port") == 9


def test_unwritable_database_dir_exits_cleanly(monkeypatch, tmp_path):
    absent_dir = tmp_path / "absent-dir"
    cfg = tmp_path / "config.yaml"
    cfg.write_text(f"memory:\n  database_path: {absent_dir / 'inf.db'}\n")
    monkeypatch.delenv("INFINITUM_CONFIG", raising=False)
    monkeypatch.setattr(sys, "argv", ["infinitum", "serve", "--config", str(cfg)])
    _patch_uvicorn(monkeypatch)

    with pytest.raises(SystemExit) as exc:
        main()

    assert "database directory missing or not writable" in str(exc.value)

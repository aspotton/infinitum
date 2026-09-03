from pathlib import Path

import pytest

import context_runtime
import infinitum
from context_runtime.app import create_app as legacy_create_app
from infinitum.app import create_app
from infinitum.config import load_config


def test_infinitum_is_primary_package_and_legacy_namespace_still_imports():
    assert infinitum.__version__ == "0.2.1"
    assert context_runtime.__version__ == infinitum.__version__
    assert legacy_create_app is create_app


def test_infinitum_config_environment_variable_is_canonical(monkeypatch, tmp_path: Path):
    cfg_path = tmp_path / "infinitum.yaml"
    cfg_path.write_text("server:\n  port: 9999\n")
    monkeypatch.setenv("INFINITUM_CONFIG", str(cfg_path))
    monkeypatch.setenv("CONTEXT_RUNTIME_CONFIG", str(tmp_path / "ignored.yaml"))

    cfg = load_config()
    assert cfg.server.port == 9999


def test_legacy_config_environment_variable_remains_supported(monkeypatch, tmp_path: Path):
    cfg_path = tmp_path / "legacy.yaml"
    cfg_path.write_text("server:\n  port: 9998\n")
    monkeypatch.delenv("INFINITUM_CONFIG", raising=False)
    monkeypatch.setenv("CONTEXT_RUNTIME_CONFIG", str(cfg_path))

    cfg = load_config()
    assert cfg.server.port == 9998


def test_unconfigured_upgrade_reuses_legacy_default_database(monkeypatch, tmp_path: Path):
    monkeypatch.delenv("INFINITUM_CONFIG", raising=False)
    monkeypatch.delenv("CONTEXT_RUNTIME_CONFIG", raising=False)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "context-runtime.db").touch()

    cfg = load_config()
    assert cfg.memory.database_path == "context-runtime.db"

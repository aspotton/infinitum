import importlib.metadata
import re
import tempfile
from pathlib import Path

import httpx
import pytest

import context_runtime
import infinitum
from context_runtime.app import create_app as legacy_create_app
from infinitum.app import create_app
from infinitum.config import AppConfig, load_config
from infinitum.runtime import build_runtime


def test_infinitum_is_primary_package_and_legacy_namespace_still_imports():
    assert re.fullmatch(r"\d+\.\d+\.\w+", infinitum.__version__)
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


async def test_health_version_matches_package_version():
    with tempfile.TemporaryDirectory() as tmp:
        # ASGITransport skips lifespan, so wire the runtime manually
        # (pattern copied from tests/test_learning_defer.py::_proxy_app).
        cfg = AppConfig()
        cfg.memory.database_path = f"{tmp}/runtime.db"
        cfg.learning.enabled = False
        cfg.upstream.passthrough_authorization = False
        app = create_app(cfg)
        rt = await build_runtime(cfg)
        app.state.runtime = rt
        try:
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(
                transport=transport, base_url="http://infinitum.test"
            ) as client:
                resp = await client.get("/health")
                assert resp.json()["version"] == infinitum.__version__
        finally:
            await rt.upstream.close()
            await rt.db.close()


def test_openapi_version_matches_package_version():
    assert (
        create_app(AppConfig()).openapi()["info"]["version"] == infinitum.__version__
    )


def test_installed_distribution_version_matches_package_version():
    # Intentionally fails (not skips) when a stale editable install disagrees
    # with the source version: rerun `uv pip install -e .` after version bumps.
    try:
        installed = importlib.metadata.version("infinitum")
    except importlib.metadata.PackageNotFoundError:
        pytest.skip("infinitum not installed as a distribution")
    assert installed == infinitum.__version__

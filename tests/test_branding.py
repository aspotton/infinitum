import importlib.metadata
import re
import tempfile
from pathlib import Path

import httpx
import pytest

import infinitum
from infinitum.app import create_app
from infinitum.config import AppConfig, load_config
from infinitum.runtime import build_runtime


def test_infinitum_config_environment_variable_is_canonical(monkeypatch, tmp_path: Path):
    cfg_path = tmp_path / "infinitum.yaml"
    cfg_path.write_text("server:\n  port: 9999\n")
    monkeypatch.setenv("INFINITUM_CONFIG", str(cfg_path))

    cfg = load_config()
    assert cfg.server.port == 9999


async def test_health_version_matches_package_version():
    assert re.fullmatch(r"\d+\.\d+\.\w+", infinitum.__version__)
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

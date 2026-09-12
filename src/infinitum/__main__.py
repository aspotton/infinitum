from __future__ import annotations

import argparse
import os
from pathlib import Path

import uvicorn

from .config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Infinitum OpenAI-compatible memory proxy")
    sub = parser.add_subparsers(dest="command")
    serve = sub.add_parser("serve", help="run the HTTP service")
    serve.add_argument(
        "--config",
        default=os.getenv("INFINITUM_CONFIG"),
        help="YAML config path",
    )
    args = parser.parse_args()
    if args.command not in {None, "serve"}:
        parser.error("unknown command")
    cfg_path = getattr(args, "config", None)
    try:
        cfg = load_config(cfg_path)
    except FileNotFoundError:
        raise SystemExit(
            f"infinitum: config file not found: {cfg_path} (see docs/CONFIGURATION.md)"
        ) from None
    if cfg_path:
        os.environ["INFINITUM_CONFIG"] = cfg_path
    db_dir = Path(cfg.memory.database_path).parent
    if not os.access(db_dir, os.W_OK):
        raise SystemExit(
            f"infinitum: database directory missing or not writable: {db_dir} "
            "(see docs/CONFIGURATION.md)"
        ) from None
    uvicorn.run("infinitum.app:create_app", factory=True, host=cfg.server.host, port=cfg.server.port, log_level=cfg.server.log_level)


if __name__ == "__main__":
    main()

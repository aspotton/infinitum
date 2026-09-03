from __future__ import annotations

import argparse
import os

import uvicorn

from .config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Infinitum OpenAI-compatible memory proxy")
    sub = parser.add_subparsers(dest="command")
    serve = sub.add_parser("serve", help="run the HTTP service")
    serve.add_argument(
        "--config",
        default=os.getenv("INFINITUM_CONFIG") or os.getenv("CONTEXT_RUNTIME_CONFIG"),
        help="YAML config path",
    )
    args = parser.parse_args()
    if args.command not in {None, "serve"}:
        parser.error("unknown command")
    cfg = load_config(getattr(args, "config", None))
    if getattr(args, "config", None):
        os.environ["INFINITUM_CONFIG"] = args.config
    uvicorn.run("infinitum.app:create_app", factory=True, host=cfg.server.host, port=cfg.server.port, log_level=cfg.server.log_level)


if __name__ == "__main__":
    main()

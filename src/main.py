from __future__ import annotations

import argparse
import json
import os

import uvicorn

from src.api.server import create_app
from src.config import load_config
from src.rag.pipeline import RagPipeline


def _build_index(config_path: str) -> int:
    cfg = load_config(config_path)
    result = RagPipeline(cfg).build_or_update_index()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def _serve(config_path: str, host: str, port: int, reload: bool) -> int:
    os.environ["MOBILERAG_CONFIG"] = config_path
    uvicorn.run(
        "src.api.server:create_app",
        factory=True,
        host=host,
        port=port,
        reload=reload,
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(prog="mobilerag")
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", help="Start the API server")
    serve.add_argument("--config", default="configs/mobile_rag.yaml")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--reload", action="store_true")

    build = sub.add_parser("build-index", help="Build or update the RAG index")
    build.add_argument("--config", default="configs/mobile_rag.yaml")

    args = parser.parse_args()

    if args.command == "serve":
        return _serve(args.config, args.host, args.port, args.reload)
    if args.command == "build-index":
        return _build_index(args.config)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

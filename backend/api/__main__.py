"""python -m backend.api [--host 127.0.0.1] [--port 8000]      (VAH_DEMO=1: online demo, backend/api/demo.py)"""
import argparse
import logging

import uvicorn

from .. import config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    factory = "backend.api.demo:create_demo_app" if config.DEMO.enabled else "backend.api.app:create_app"
    uvicorn.run(factory, factory=True, host=args.host, port=args.port)


if __name__ == "__main__":
    main()

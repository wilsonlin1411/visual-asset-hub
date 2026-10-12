"""python -m backend.api [--host 127.0.0.1] [--port 8000]"""
import argparse
import logging

import uvicorn


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    uvicorn.run("backend.api.app:create_app", factory=True, host=args.host, port=args.port)


if __name__ == "__main__":
    main()

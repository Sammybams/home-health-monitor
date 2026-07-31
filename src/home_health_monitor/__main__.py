from __future__ import annotations

import argparse

from .server import run


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the home health monitor")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8080, type=int)
    parser.add_argument("--model", default="artifacts/model.json")
    parser.add_argument("--max-body-bytes", default=4 * 1024 * 1024, type=int)
    args = parser.parse_args()
    run(args.host, args.port, args.model, args.max_body_bytes)


if __name__ == "__main__":
    main()

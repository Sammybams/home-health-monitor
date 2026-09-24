from __future__ import annotations

import argparse

from .gateway.server import run


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the home health gateway")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8080, type=int)
    parser.add_argument("--database", default="data/gateway.db")
    parser.add_argument("--model", default="artifacts/gateway/model.tflite")
    parser.add_argument(
        "--model-metadata", default="artifacts/gateway/model-metadata.json"
    )
    parser.add_argument("--max-body-bytes", default=64 * 1024, type=int)
    parser.add_argument("--vector-model", default="artifacts/real-ppg-v2/model.tflite")
    parser.add_argument(
        "--vector-model-metadata",
        default="artifacts/real-ppg-v2/model-metadata.json",
    )
    args = parser.parse_args()
    run(
        args.host,
        args.port,
        database_path=args.database,
        model_path=args.model,
        metadata_path=args.model_metadata,
        max_body_bytes=args.max_body_bytes,
        vector_model_path=args.vector_model,
        vector_metadata_path=args.vector_model_metadata,
    )


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .baseline import baseline_to_dict, build_baseline
from .contracts import InputError, parse_request


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a personal baseline from healthy daily windows")
    parser.add_argument("input", type=Path, help="JSON Lines file containing 7-30 healthy request windows")
    parser.add_argument("output", type=Path, help="Destination baseline JSON file")
    args = parser.parse_args()

    requests = []
    try:
        with args.input.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    requests.append(parse_request(json.loads(line)))
                except (json.JSONDecodeError, InputError) as exc:
                    raise InputError(f"invalid baseline row {line_number}: {exc}") from exc
        profile = build_baseline(requests)
    except (OSError, InputError) as exc:
        parser.error(str(exc))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(baseline_to_dict(profile), indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()

"""Read-only manifest/event polling with explicit bounded sleeps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any


TERMINAL = {"completed", "failed"}


def read_status(manifest_path: Path, events_path: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    latest_event = None
    if events_path.exists():
        lines = [line for line in events_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if lines:
            latest_event = json.loads(lines[-1])
    return {"manifest": manifest, "latest_event": latest_event}


def watch(manifest: Path, events: Path, *, interval: float, max_polls: int | None) -> str:
    if interval <= 0.0 or (max_polls is not None and max_polls <= 0):
        raise ValueError("Interval and max polls must be positive.")
    polls = 0
    while max_polls is None or polls < max_polls:
        state = read_status(manifest, events)
        print(json.dumps(state, sort_keys=True), flush=True)
        polls += 1
        status = state["manifest"].get("status")
        if status in TERMINAL:
            return str(status)
        if max_polls is None or polls < max_polls:
            time.sleep(interval)
    return str(state["manifest"].get("status"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=60.0)
    parser.add_argument("--max-polls", type=int)
    args = parser.parse_args()
    watch(args.manifest, args.events, interval=args.interval, max_polls=args.max_polls)


if __name__ == "__main__":
    main()


#!/usr/bin/env python
"""One read-only Run 013 snapshot; the controller owns bounded poll timing."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

from run_config import RUN_DIR


def snapshot(root: Path = RUN_DIR) -> dict:
    workers = {}
    progress_root = root / "artifacts" / "workers"
    if progress_root.exists():
        for path in sorted(progress_root.glob("*/progress.json")):
            workers[path.parent.name] = json.loads(path.read_text(encoding="utf-8"))
    attempts = []
    attempts_root = root / "artifacts" / "attempts"
    if attempts_root.exists():
        for attempt in sorted(path for path in attempts_root.iterdir() if path.is_dir()):
            manifest_path = attempt / "manifest.json"
            events_path = attempt / "events.jsonl"
            manifest = (
                json.loads(manifest_path.read_text(encoding="utf-8"))
                if manifest_path.exists()
                else {}
            )
            last_event = None
            if events_path.exists():
                with events_path.open("rb") as handle:
                    lines = [line for line in handle.read().splitlines() if line]
                last_event = json.loads(lines[-1]) if lines else None
            attempts.append({
                "attempt_id": attempt.name,
                "status": manifest.get("status"),
                "condition_id": manifest.get("condition", {}).get("id"),
                "last_event": last_event,
            })
    return {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "workers": workers,
        "attempts": attempts,
    }


if __name__ == "__main__":
    print(json.dumps(snapshot(), indent=2, sort_keys=True))

#!/usr/bin/env python3
"""Emit one read-only Run-022 worker/GPU monitoring snapshot."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def read(path: Path) -> str | None:
    return path.read_text(encoding="utf-8").strip() if path.is_file() else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=Path("/workspace/run022-output"))
    parser.add_argument("--attempt-id")
    parser.add_argument("--log-lines", type=int, default=30)
    args = parser.parse_args()
    control_root = args.output_root / "control"
    attempts = sorted(path for path in control_root.glob("*") if path.is_dir())
    if args.attempt_id:
        control = control_root / args.attempt_id
    elif attempts:
        control = attempts[-1]
    else:
        raise FileNotFoundError(f"No attempts under {control_root}")
    pid_text = read(control / "worker.pid")
    alive = False
    if pid_text:
        try:
            os.kill(int(pid_text), 0)
            alive = True
        except (OSError, ValueError):
            pass
    log_path = control / "worker.log"
    log_lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-args.log_lines:] if log_path.is_file() else []
    attempt_dir = args.output_root / "attempts" / control.name
    progress_path = attempt_dir / "progress.json"
    progress = json.loads(progress_path.read_text(encoding="utf-8")) if progress_path.is_file() else None
    gpu = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=timestamp,name,utilization.gpu,memory.used,memory.total,power.draw",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    snapshot = {
        "checked_utc": datetime.now(timezone.utc).isoformat(),
        "attempt_id": control.name,
        "pid": int(pid_text) if pid_text and pid_text.isdigit() else None,
        "worker_alive": alive,
        "status": read(control / "status.txt"),
        "started_utc": read(control / "started-utc.txt"),
        "finished_utc": read(control / "finished-utc.txt"),
        "exit_code": read(control / "exit-code.txt"),
        "progress": progress,
        "result_archive": read(control / "result-archive.txt"),
        "gpu": gpu.stdout.strip() if gpu.returncode == 0 else gpu.stderr.strip(),
        "log_tail": log_lines,
    }
    print(json.dumps(snapshot, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

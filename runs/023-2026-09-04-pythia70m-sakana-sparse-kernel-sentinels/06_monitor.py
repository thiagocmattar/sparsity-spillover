#!/usr/bin/env python3
"""Emit one read-only Run-023 progress, ETC, GPU, and cost snapshot."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path


def read(path: Path) -> str | None:
    return path.read_text(encoding="utf-8", errors="replace").strip() if path.is_file() else None


def parse_utc(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=Path("/workspace/run023-output"))
    parser.add_argument("--attempt-id")
    parser.add_argument("--log-lines", type=int, default=35)
    parser.add_argument("--hourly-price", type=float)
    parser.add_argument("--storage-hourly-price", type=float, default=0.0)
    parser.add_argument("--account-balance", type=float)
    parser.add_argument("--guard-hours", type=float, default=4.0)
    parser.add_argument("--pod-created-utc")
    args = parser.parse_args()
    control_root = args.output_root / "control"
    attempts = sorted(path for path in control_root.glob("*") if path.is_dir())
    control = control_root / args.attempt_id if args.attempt_id else (attempts[-1] if attempts else None)
    if control is None or not control.is_dir():
        raise FileNotFoundError(f"No Run-023 attempt under {control_root}")
    pid_text = read(control / "worker.pid")
    alive = False
    if pid_text:
        try:
            os.kill(int(pid_text), 0)
            alive = True
        except (OSError, ValueError):
            pass
    attempt_dir = args.output_root / "attempts" / control.name
    progress_path = attempt_dir / "progress.json"
    progress = json.loads(progress_path.read_text(encoding="utf-8")) if progress_path.is_file() else None
    now = datetime.now(timezone.utc)
    started = parse_utc(args.pod_created_utc) or parse_utc(read(control / "started-utc.txt"))
    elapsed_hours = (now - started).total_seconds() / 3600 if started else None
    etc_seconds = None
    etc_scope = None
    if progress:
        phase_completion = progress.get("phase_projected_completion_unix")
        if isinstance(phase_completion, (int, float)) and math.isfinite(phase_completion):
            etc_seconds = max(0.0, float(phase_completion) - now.timestamp())
            etc_scope = (
                "phase_projection"
                if float(phase_completion) >= now.timestamp()
                else "phase_projection_overdue"
            )
        else:
            value = progress.get("stage_etc_seconds")
            if isinstance(value, (int, float)) and math.isfinite(value) and value >= 0:
                etc_seconds = float(value)
                etc_scope = "current_stage_only"
    hourly = (
        args.hourly_price + args.storage_hourly_price
        if args.hourly_price is not None
        else None
    )
    accrued = elapsed_hours * hourly if elapsed_hours is not None and hourly is not None else None
    required = etc_seconds / 3600 * hourly if etc_seconds is not None and hourly is not None else None
    guard_remaining_hours = (
        max(0.0, args.guard_hours - elapsed_hours) if elapsed_hours is not None else None
    )
    guard_remaining_cost = (
        guard_remaining_hours * hourly
        if guard_remaining_hours is not None and hourly is not None
        else None
    )
    projected = now + timedelta(seconds=etc_seconds) if etc_seconds is not None else None
    progress_age_minutes = (
        (now.timestamp() - progress_path.stat().st_mtime) / 60 if progress_path.is_file() else None
    )
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
    log_path = control / "worker.log"
    log_tail = (
        log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-args.log_lines :]
        if log_path.is_file()
        else []
    )
    warnings = []
    if progress_age_minutes is not None and progress_age_minutes > 20 and alive:
        warnings.append("no progress update for more than 20 minutes")
    if elapsed_hours is not None and elapsed_hours >= args.guard_hours:
        warnings.append("maximum-duration guard reached")
    if (
        started is not None
        and projected is not None
        and projected > started + timedelta(hours=args.guard_hours)
        and etc_scope == "phase_projection"
    ):
        warnings.append("phase projection crosses maximum-duration guard")
    if etc_scope == "phase_projection_overdue" and alive:
        warnings.append("phase projection elapsed before worker completion; recalibration is overdue")
    balance_requirement = required if etc_scope == "phase_projection" else guard_remaining_cost
    if (
        args.account_balance is not None
        and balance_requirement is not None
        and args.account_balance < balance_requirement
    ):
        warnings.append("reported balance is below the current completion/guard requirement")
    snapshot = {
        "checked_utc": now.isoformat(),
        "attempt_id": control.name,
        "pid": int(pid_text) if pid_text and pid_text.isdigit() else None,
        "worker_alive": alive,
        "status": read(control / "status.txt"),
        "started_utc": read(control / "started-utc.txt"),
        "finished_utc": read(control / "finished-utc.txt"),
        "exit_code": read(control / "exit-code.txt"),
        "progress": progress,
        "progress_age_minutes": progress_age_minutes,
        "etc_seconds": etc_seconds,
        "etc_scope": etc_scope,
        "projected_completion_utc": projected.isoformat() if projected else None,
        "cost": {
            "gpu_hourly": args.hourly_price,
            "storage_hourly": args.storage_hourly_price,
            "elapsed_hours": elapsed_hours,
            "accrued_estimate": accrued,
            "required_to_projection": required,
            "guard_remaining_hours": guard_remaining_hours,
            "guard_remaining_cost": guard_remaining_cost,
            "balance_requirement": balance_requirement,
            "account_balance": args.account_balance,
            "guard_hours": args.guard_hours,
        },
        "result_archive": read(control / "result-archive.txt"),
        "gpu": gpu.stdout.strip() if gpu.returncode == 0 else gpu.stderr.strip(),
        "warnings": warnings,
        "log_tail": log_tail,
    }
    print(json.dumps(snapshot, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

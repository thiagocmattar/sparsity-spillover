"""Small immutable-attempt lifecycle and transfer inventory."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, TextIO
from uuid import uuid4

import yaml

from .data import file_sha256


@dataclass
class Attempt:
    run_dir: Path
    attempt_dir: Path
    attempt_id: str
    config: dict[str, Any]
    manifest: dict[str, Any]
    _terminal: bool = field(default=False, init=False, repr=False)

    def append_event(self, event: Mapping[str, Any]) -> None:
        if self._terminal:
            raise RuntimeError("Cannot append to a terminal attempt.")
        path = self.attempt_dir / "events.jsonl"
        with path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(dict(event), sort_keys=True))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())

    def complete(
        self,
        *,
        metrics: Mapping[str, Any],
        predictions: Sequence[Mapping[str, Any]] = (),
        manifest_updates: Mapping[str, Any] | None = None,
    ) -> Path:
        self._require_running()
        _atomic_json(self.attempt_dir / "metrics.json", metrics)
        _atomic_jsonl(self.attempt_dir / "predictions.jsonl", predictions)
        terminal = {**self.manifest, **deepcopy(dict(manifest_updates or {}))}
        terminal.update(status="completed", finished_at=_utc_now())
        _atomic_json(self.attempt_dir / "manifest.json", terminal)
        self.manifest = terminal
        self._terminal = True
        return self.attempt_dir

    def fail(self, error: BaseException) -> Path:
        self._require_running()
        terminal = {
            **self.manifest,
            "status": "failed",
            "finished_at": _utc_now(),
            "failure": {"type": type(error).__qualname__, "message": str(error)},
        }
        _atomic_json(self.attempt_dir / "manifest.json", terminal)
        self.manifest = terminal
        self._terminal = True
        return self.attempt_dir

    def _require_running(self) -> None:
        if self._terminal or self.manifest.get("status") != "running":
            raise RuntimeError("Attempt is already terminal.")


def start_attempt(
    run_dir: str | Path,
    *,
    config: Mapping[str, Any],
    command: str,
    mode: str,
    extra_identity: Mapping[str, Any] | None = None,
) -> Attempt:
    root = Path(run_dir).resolve()
    attempts = root / "artifacts" / "attempts"
    attempts.mkdir(parents=True, exist_ok=True)
    sequence = _next_sequence(attempts)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    attempt_id = f"{sequence:03d}-{stamp}-{uuid4().hex[:8]}"
    attempt_dir = attempts / attempt_id
    attempt_dir.mkdir(exist_ok=False)

    snapshot = deepcopy(dict(config))
    _atomic_yaml(attempt_dir / "config.yaml", snapshot)
    manifest = {
        "schema_version": 1,
        "attempt_id": attempt_id,
        "status": "running",
        "mode": str(mode),
        "command": str(command),
        "started_at": _utc_now(),
        "config_sha256": config_sha256(snapshot),
        "code": git_identity(root),
        "environment": environment_identity(),
        **deepcopy(dict(extra_identity or {})),
    }
    _atomic_json(attempt_dir / "manifest.json", manifest)
    (attempt_dir / "events.jsonl").touch(exist_ok=False)
    return Attempt(root, attempt_dir, attempt_id, snapshot, manifest)


@contextmanager
def attempt_lifecycle(
    run_dir: str | Path,
    *,
    config: Mapping[str, Any],
    command: str,
    mode: str,
    extra_identity: Mapping[str, Any] | None = None,
) -> Iterator[Attempt]:
    attempt = start_attempt(
        run_dir,
        config=config,
        command=command,
        mode=mode,
        extra_identity=extra_identity,
    )
    try:
        yield attempt
    except BaseException as error:
        if not attempt._terminal:
            attempt.fail(error)
        raise
    if not attempt._terminal:
        error = RuntimeError("Attempt lifecycle exited without complete() or fail().")
        attempt.fail(error)
        raise error


def build_transfer_inventory(
    root: str | Path,
    *,
    exclude: Sequence[str] = (),
) -> dict[str, Any]:
    base = Path(root).resolve()
    excluded = set(exclude)
    rows = []
    for path in sorted(item for item in base.rglob("*") if item.is_file()):
        relative = path.relative_to(base).as_posix()
        if relative in excluded or path.name == "transfer_inventory.json":
            continue
        rows.append(
            {"path": relative, "bytes": path.stat().st_size, "sha256": file_sha256(path)}
        )
    return {
        "schema_version": 1,
        "root": base.name,
        "files": rows,
        "total_bytes": sum(row["bytes"] for row in rows),
    }


def verify_transfer_inventory(root: str | Path, inventory: Mapping[str, Any]) -> None:
    base = Path(root).resolve()
    rows = inventory.get("files")
    if not isinstance(rows, list):
        raise ValueError("Transfer inventory has no file list.")
    for row in rows:
        path = (base / row["path"]).resolve()
        try:
            path.relative_to(base)
        except ValueError as error:
            raise ValueError("Transfer inventory path escapes its root.") from error
        if path.stat().st_size != row["bytes"] or file_sha256(path) != row["sha256"]:
            raise ValueError(f"Transferred artifact identity mismatch: {row['path']}")


def config_sha256(config: Mapping[str, Any]) -> str:
    payload = json.dumps(dict(config), sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def git_identity(cwd: Path) -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=cwd, check=True, capture_output=True, text=True
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"], cwd=cwd, check=True, capture_output=True, text=True
            ).stdout.strip()
        )
        return {"git_commit": commit, "git_dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"git_commit": None, "git_dirty": None}


def environment_identity() -> dict[str, Any]:
    identity = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    try:
        import torch

        identity.update(
            torch=torch.__version__,
            cuda_runtime=torch.version.cuda,
            cuda_available=torch.cuda.is_available(),
            devices=[torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())],
        )
    except ImportError:
        identity["torch"] = None
    try:
        import transformers

        identity["transformers"] = transformers.__version__
    except ImportError:
        identity["transformers"] = None
    return identity


def _next_sequence(attempts: Path) -> int:
    values = []
    for path in attempts.iterdir():
        if path.is_dir() and path.name[:3].isdigit():
            values.append(int(path.name[:3]))
    return max(values, default=0) + 1


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    with _atomic_text(path) as handle:
        json.dump(dict(value), handle, indent=2, sort_keys=True)
        handle.write("\n")


def _atomic_yaml(path: Path, value: Mapping[str, Any]) -> None:
    with _atomic_text(path) as handle:
        yaml.safe_dump(dict(value), handle, sort_keys=False)


def _atomic_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with _atomic_text(path) as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True))
            handle.write("\n")


@contextmanager
def _atomic_text(path: Path) -> Iterator[TextIO]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            yield handle
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


"""Small run-local identity, safe-path, and durable-record helpers."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[1]


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inside(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Path escapes its declared root: {relative}")
    return path


def record(path, root=ROOT):
    path = Path(path)
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"Expected regular input file: {path}")
    return {"path": path.resolve().relative_to(Path(root).resolve()).as_posix(),
            "bytes": path.stat().st_size, "sha256": sha256(path)}


def verify_record(row, root=ROOT):
    path = inside(root, row["path"])
    if record(path, root) != row:
        raise ValueError(f"Input identity mismatch: {row['path']}")
    return path


def config():
    value = read_json(RUN / "config.json")
    inp, cal, budget = value["inputs"], value["calibration"], value["budget"]
    for name in ("sequence_length", "validation_blocks", "development_blocks"):
        if type(inp[name]) is not int or inp[name] <= 0:
            raise ValueError(f"Invalid input count: {name}")
    for name in ("timing_inputs", "timing_passes", "max_worker_seconds"):
        if type(cal[name]) is not int or cal[name] <= 0:
            raise ValueError(f"Invalid calibration count: {name}")
    if cal["timing_inputs"] > inp["development_blocks"]:
        raise ValueError("Timing inputs exceed development coverage")
    if not 0 < budget["pilot_usd"] <= budget["total_usd"]:
        raise ValueError("Pilot must fit the total budget")
    for key, number in budget.items():
        if not isinstance(number, (float, int)) or not math.isfinite(number) or number < 0:
            raise ValueError(f"Invalid monetary value: {key}")
    return value


def select_conditions(manifest, phase):
    rows = manifest["checkpoints"]
    if phase == "primitive":
        return []
    if phase == "calibration":
        wanted = config()["calibration"]["conditions"]
        return [next(row for row in rows if row["id"] == name) for name in wanted]
    if phase == "development":
        return [row for row in rows if row["partition"] == "development"]
    if phase == "final":
        return list(rows)
    raise ValueError(f"Unknown phase: {phase}")

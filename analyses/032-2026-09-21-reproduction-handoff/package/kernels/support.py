"""Relocatable paths for the unchanged final CUDA implementations."""

from pathlib import Path
import hashlib
import importlib.util
import json
import sys

ROOT = Path(__file__).resolve().parent
RUN = ROOT / "14m"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def record(path):
    path = Path(path)
    return dict(
        path=path.relative_to(ROOT).as_posix(),
        bytes=path.stat().st_size,
        sha256=sha256(path),
    )


def verify_record(row):
    path = ROOT / row["path"]
    if not path.resolve().is_relative_to(ROOT.resolve()):
        raise ValueError("Unsafe dependency path")
    if sha256(path) != row["sha256"] or path.stat().st_size != row["bytes"]:
        raise ValueError(f"Changed dependency: {path}")
    return path


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m

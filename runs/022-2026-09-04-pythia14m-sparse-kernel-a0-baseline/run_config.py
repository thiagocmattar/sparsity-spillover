"""Run-022 configuration loading and immutable-input checks."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


RUN_DIR = Path(__file__).resolve().parent
REPO_ROOT = RUN_DIR.parents[1]


def load_config(path: Path | None = None) -> dict[str, Any]:
    config_path = path or RUN_DIR / "config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Run configuration must be a mapping.")
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    model = config["model"]
    arch = model["architecture"]
    validation = config["validation"]
    measurement = config["measurement"]
    if model["topology_id"] != "A0" or model["gate"] != "gelu":
        raise ValueError("Run 022 is the fixed A0/GELU baseline only.")
    if (arch["hidden_size"], arch["intermediate_size"], arch["layers"]) != (128, 512, 6):
        raise ValueError("Unexpected Pythia-14M architecture.")
    if validation["complete_blocks"] * arch["sequence_length"] != validation["evaluated_tokens"]:
        raise ValueError("Complete validation coverage is inconsistent.")
    if validation["source_tokens"] - validation["evaluated_tokens"] != validation["excluded_tail_tokens"]:
        raise ValueError("Validation tail is inconsistent.")
    if measurement["tile_width"] != 256 or arch["intermediate_size"] % 256:
        raise ValueError("TwELL occupancy accounting requires 256-wide tiles.")
    if measurement["raw_ell_alignment"] <= 0:
        raise ValueError("ELL alignment must be positive.")


def repo_path(relative: str | Path) -> Path:
    result = (REPO_ROOT / relative).resolve()
    try:
        result.relative_to(REPO_ROOT.resolve())
    except ValueError as error:
        raise ValueError(f"Path escapes repository: {relative}") from error
    return result


def sha256_file(path: Path, *, chunk_bytes: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def verify_file(path: Path, expected_sha256: str, expected_bytes: int | None = None) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    actual_bytes = path.stat().st_size
    actual_sha256 = sha256_file(path)
    if expected_bytes is not None and actual_bytes != expected_bytes:
        raise ValueError(f"Size mismatch for {path}: {actual_bytes} != {expected_bytes}")
    if actual_sha256 != expected_sha256:
        raise ValueError(f"SHA-256 mismatch for {path}: {actual_sha256} != {expected_sha256}")
    return {"path": display_path(path), "bytes": actual_bytes, "sha256": actual_sha256}


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")

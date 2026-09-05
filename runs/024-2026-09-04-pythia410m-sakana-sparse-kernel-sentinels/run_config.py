"""Run-024 configuration and immutable-input helpers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


RUN_DIR = Path(__file__).resolve().parent
REPO_ROOT = RUN_DIR.parents[1]
EXPECTED_SENTINELS = (
    "a0-gelu",
    "a1h-relu",
    "a4-ol1-kappa-0",
    "a4-ol1-kappa-0p5",
    "a7-ol1-kappa-0",
    "a7-ol1-kappa-0p5",
)
LINEAR_OPERATION_SITES = {
    "qkv_projection": "a",
    "mlp_w1": "m",
    "mlp_w2": "h",
    "attention_output_projection": "z",
}


def load_config(path: Path | None = None) -> dict[str, Any]:
    config_path = path or RUN_DIR / "config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Run configuration must be a mapping.")
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    architecture = config["model"]["architecture"]
    actual_architecture = (
        architecture["hidden_size"],
        architecture["intermediate_size"],
        architecture["layers"],
        architecture["attention_heads"],
        architecture["head_size"],
    )
    if actual_architecture != (1024, 4096, 24, 16, 64):
        raise ValueError(f"Run 024 requires fixed Pythia-410M architecture, got {actual_architecture}.")
    if architecture["attention_heads"] * architecture["head_size"] != architecture["hidden_size"]:
        raise ValueError("Attention head decomposition does not equal hidden size.")

    conditions = config["conditions"]
    ids = tuple(condition["id"] for condition in conditions)
    if ids != EXPECTED_SENTINELS:
        raise ValueError(f"Unexpected sentinel order: {ids}")
    if tuple(config["condition_partition"]["sentinel"]) != EXPECTED_SENTINELS:
        raise ValueError("The sentinel partition must exactly match the six confirmed conditions.")

    for condition in conditions:
        operations = condition["linear_operations"]
        if not operations or any(operation not in LINEAR_OPERATION_SITES for operation in operations):
            raise ValueError(f"Invalid linear coverage for {condition['id']}: {operations}")
        if condition["step"] in {"A0", "A1-H"} and operations != ["mlp_w2"]:
            raise ValueError("A0/A1-H may cover only h -> W2.")
        if condition["step"] == "A4-OL1" and condition["attention_microbenchmark"]:
            raise ValueError("A4 does not activate the attention microbenchmark.")
        if condition["step"] == "A7-OL1" and not condition["attention_microbenchmark"]:
            raise ValueError("Every A7 condition must retain the attention microbenchmark.")
        if condition["step"] == "A7-OL1" and condition["topology_id"] != "A7-Z-POST":
            raise ValueError("A7 must use post-RoPE q/k topology.")

    validation = config["validation"]
    if validation["complete_blocks"] * architecture["sequence_length"] != validation["evaluated_tokens"]:
        raise ValueError("Complete validation block coverage is inconsistent.")
    if validation["source_tokens"] - validation["evaluated_tokens"] != validation["excluded_tail_tokens"]:
        raise ValueError("Validation tail is inconsistent.")
    if validation["source_reproduction_batch_size"] != 4:
        raise ValueError("Source loss reproduction must match Run 019's batch size four.")
    if validation["runtime_equivalence_batch_size"] != 1:
        raise ValueError("Run 024's memory-safe dense/sparse equivalence batch must be one.")
    if validation["logical_opportunity_batch_size"] != 1:
        raise ValueError("Logical opportunities must match Run 019's batch-one diagnostic.")

    measurement = config["measurement"]
    batches = measurement["full_model_batches"]
    if not batches or batches[0] != 1 or any(batch not in {1, 32} for batch in batches):
        raise ValueError("Full-model batches must start with primary batch one and may only add batch 32.")
    if measurement["runtime_dtype"] != "bfloat16":
        raise ValueError("The derived ELL path is fixed to BF16.")
    if measurement["tile_width"] != 256:
        raise ValueError("TwELL occupancy summaries require 256-wide tiles.")
    if config["upstream"]["derivative_label"] != "Sakana-derived":
        raise ValueError("The modified path must not be labeled as official upstream.")


def repo_path(relative: str | Path) -> Path:
    result = (REPO_ROOT / relative).resolve()
    try:
        result.relative_to(REPO_ROOT.resolve())
    except ValueError as error:
        raise ValueError(f"Path escapes repository: {relative}") from error
    return result


def condition_by_id(config: dict[str, Any], condition_id: str) -> dict[str, Any]:
    matches = [condition for condition in config["conditions"] if condition["id"] == condition_id]
    if len(matches) != 1:
        raise ValueError(f"Unknown or duplicate condition: {condition_id}")
    return matches[0]


def conditions_for_phase(config: dict[str, Any], phase: str) -> list[dict[str, Any]]:
    if phase != "sentinel":
        raise ValueError(f"Run 024 has only the confirmed sentinel phase, not {phase!r}.")
    return [condition_by_id(config, condition_id) for condition_id in EXPECTED_SENTINELS]


def checkpoint_path(config: dict[str, Any], condition: dict[str, Any]) -> Path:
    relative = (
        Path(config["source_run"]["root"])
        / condition["attempt_id"]
        / "checkpoints"
        / config["source_run"]["checkpoint_step"]
    )
    return repo_path(relative)


def logical_products_path(config: dict[str, Any], condition: dict[str, Any]) -> Path:
    return repo_path(
        Path(config["source_run"]["root"])
        / condition["attempt_id"]
        / "diagnostics"
        / "logical_products.json"
    )


def checkpoint_file_records(config: dict[str, Any], condition: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        **config["source_run"]["shared_checkpoint_files"],
        "config.json": condition["config_file"],
        "model.safetensors": condition["model_file"],
    }


def sha256_file(path: Path, *, chunk_bytes: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def verify_file(path: Path, expected_sha256: str, expected_bytes: int | None = None) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(path)
    actual_bytes = path.stat().st_size
    if expected_bytes is not None and actual_bytes != expected_bytes:
        raise ValueError(f"Size mismatch for {path}: {actual_bytes} != {expected_bytes}")
    actual_sha256 = sha256_file(path)
    if actual_sha256 != expected_sha256:
        raise ValueError(f"SHA-256 mismatch for {path}: {actual_sha256} != {expected_sha256}")
    return {"path": display_path(path), "bytes": actual_bytes, "sha256": actual_sha256}


def display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


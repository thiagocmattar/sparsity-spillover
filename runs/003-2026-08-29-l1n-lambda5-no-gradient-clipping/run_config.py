"""Approved Run 003 configuration, cache identities, and schedule helpers."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import yaml

from sparsity_research.data import (
    build_training_schedule,
    file_sha256,
    require_full_minipile_validation,
    verify_token_cache,
)
from sparsity_research.pressure import parse_pressure_config
from sparsity_research.sites import resolve_topology_and_gate


RUN_DIR = Path(__file__).resolve().parent
REPO_ROOT = RUN_DIR.parents[1]
DEFAULT_CONFIG = RUN_DIR / "config.yaml"
TRAIN_CACHE_SHA256 = "da82a2ea2e0080c7fd681c7a93b07d3d9ff3d5357a8640895a82d536a1eaf97c"
TRAIN_CACHE_TOKENS = 1_491_711_416
TRAIN_CACHE_DOCUMENTS = 1_000_000
EXPECTED_ACTIVATIONS = ("gelu", "relu")


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Run config must be a mapping.")
    validate_config(config)
    return config


def validate_config(config: Mapping[str, Any]) -> None:
    if mapping(config, "model").get("initialization") != "random":
        raise ValueError("Pythia pretraining requires random initialization.")

    conditions = mapping(config, "conditions")
    if tuple(conditions.get("activations", ())) != EXPECTED_ACTIVATIONS:
        raise ValueError("Conditions must contain GeLU then ReLU.")
    if conditions.get("pressure_method") != "l1_naive":
        raise ValueError("Run 003 requires naive L1 pressure.")
    if conditions.get("pressure_site") != "h" or float(
        conditions.get("pressure_weight", -1.0)
    ) != 5.0:
        raise ValueError("Run 003 requires lambda=5 pressure only at h.")

    training = mapping(config, "training")
    for key in (
        "max_steps",
        "global_batch_size",
        "micro_batch_size",
        "gradient_accumulation_steps",
    ):
        value = training.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"training.{key} must be a positive integer.")
    if int(training["micro_batch_size"]) * int(
        training["gradient_accumulation_steps"]
    ) != int(training["global_batch_size"]):
        raise ValueError("Microbatch times accumulation must equal global batch.")
    for key in (
        "peak_learning_rate",
        "adamw_eps",
        "warmup_fraction",
        "minimum_learning_rate_ratio",
        "target_cohort_seconds",
        "planning_cohort_seconds",
    ):
        value = float(training.get(key, math.nan))
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"training.{key} must be positive and finite.")
    if training.get("gradient_clip_norm") is not None:
        raise ValueError("Run 003 must disable norm-based gradient clipping.")
    if training.get("finite_gradient_check") is not True:
        raise ValueError("Finite-gradient validation must remain enabled.")
    if training.get("optimizer") != "adamw":
        raise ValueError("Run 003 requires AdamW.")
    if training.get("device") != "cuda" or training.get("precision") != "bfloat16":
        raise ValueError("The approved local execution requires CUDA BF16.")

    data = mapping(config, "data")
    if int(data.get("sequence_length", 0)) != 2048:
        raise ValueError("The data contract requires 2,048-token sequences.")
    validation = mapping(config, "validation")
    expected_validation = {
        "documents": 500,
        "complete_sequences": 338,
        "input_tokens": 692_224,
        "excluded_tail_tokens": 1_444,
    }
    if any(
        int(validation.get(key, -1)) != value
        for key, value in expected_validation.items()
    ):
        raise ValueError("Complete MiniPile validation coverage does not match.")

    diagnostics = mapping(config, "diagnostics")
    if tuple(diagnostics.get("activation_sites", ())) != (
        "h",
        "q_post",
        "k_post",
        "v",
        "m",
        "attention_output",
    ):
        raise ValueError("Run 003 terminal diagnostic sites do not match the design.")
    if tuple(float(value) for value in diagnostics.get("near_zero_thresholds", ())) != (
        0.0,
        0.001,
        0.01,
    ):
        raise ValueError("Run 003 near-zero thresholds do not match the design.")
    if diagnostics.get("logical_products") is not False or diagnostics.get(
        "clipping_frontier"
    ) is not None:
        raise ValueError("Unapproved logical/clipping diagnostics are enabled.")

    comparison = mapping(config, "comparison")
    baseline = repo_path(str(comparison["baseline_verification"]))
    if not baseline.is_file() or file_sha256(baseline) != comparison.get(
        "baseline_verification_sha256"
    ):
        raise ValueError("The immutable Run 002 baseline verification changed.")
    if len(condition_specs(config)) != 2:
        raise ValueError("Exactly two unclipped conditions are required.")


def condition_specs(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    conditions = mapping(config, "conditions")
    weight = float(conditions["pressure_weight"])
    return [
        {
            "id": f"{activation}-l1n-5-no-clip",
            "order": index,
            "activation": activation,
            "pressure_method": "l1_naive",
            "pressure_sites": ["h"],
            "pressure_weight": weight,
            "gradient_clip_norm": None,
            "label": f"{activation} lambda=5, no clipping",
        }
        for index, activation in enumerate(EXPECTED_ACTIVATIONS, start=1)
    ]


def resolved_condition_config(
    config: Mapping[str, Any], condition: Mapping[str, Any]
) -> dict[str, Any]:
    resolved = deepcopy(dict(config))
    resolved["condition"] = deepcopy(dict(condition))
    topology_id = "A0" if condition["activation"] == "gelu" else "A1-H"
    site_gate = None if condition["activation"] == "gelu" else {"operator": "relu"}
    resolve_topology_and_gate(topology_id, site_gate)
    resolved["model"]["topology_id"] = topology_id
    resolved["model"]["site_gate"] = site_gate
    pressure = {
        "method": "l1_naive",
        "sites": ["h"],
        "weight": float(condition["pressure_weight"]),
        "step_budget": None,
        "eps": 1.0e-12,
    }
    parse_pressure_config(pressure)
    resolved["activation_pressure"] = pressure
    return resolved


def load_verified_caches(
    config: Mapping[str, Any], *, np: Any
) -> tuple[Any, Any, dict[str, Any], dict[str, Any], float]:
    from time import perf_counter

    started = perf_counter()
    data = mapping(config, "data")
    train_metadata_path = repo_path(str(data["training_metadata"]))
    validation_metadata_path = repo_path(str(data["validation_metadata"]))
    train_metadata = json.loads(train_metadata_path.read_text(encoding="utf-8"))
    validation_metadata = json.loads(
        validation_metadata_path.read_text(encoding="utf-8")
    )
    train_path = verify_token_cache(train_metadata, base=train_metadata_path.parent)
    validation_path = verify_token_cache(
        validation_metadata, base=validation_metadata_path.parent
    )
    require_training_cache(train_metadata)
    require_full_minipile_validation(validation_metadata)
    train_tokens = np.memmap(
        train_path,
        dtype=np.int32,
        mode="r",
        shape=(int(train_metadata["tokens"]),),
    )
    validation_tokens = np.memmap(
        validation_path,
        dtype=np.int32,
        mode="r",
        shape=(int(validation_metadata["tokens"]),),
    )
    return (
        train_tokens,
        validation_tokens,
        train_metadata,
        validation_metadata,
        perf_counter() - started,
    )


def build_schedule(
    config: Mapping[str, Any], train_metadata: Mapping[str, Any], *, np: Any
) -> tuple[Any, str, dict[str, Any]]:
    training = mapping(config, "training")
    return build_training_schedule(
        np,
        token_count=int(train_metadata["tokens"]),
        block_size=int(mapping(config, "data")["sequence_length"]),
        max_steps=int(training["max_steps"]),
        gradient_accumulation_steps=int(training["gradient_accumulation_steps"]),
        micro_batch_size=int(training["micro_batch_size"]),
        seed=int(mapping(config, "seeds")["data_order"]),
    )


def microbatches_for_step(
    tokens: Any,
    step_starts: Any,
    *,
    block_size: int,
    device: Any,
    torch: Any,
    np: Any,
) -> list[Any]:
    return [
        torch.as_tensor(
            np.stack(
                [
                    tokens[int(start) : int(start) + int(block_size)]
                    for start in micro_starts
                ]
            ),
            dtype=torch.long,
            device=device,
        )
        for micro_starts in step_starts
    ]


def parameter_sha256(model: Any) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        tensor = value.detach().contiguous().cpu()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(json.dumps(list(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def inventory_content_sha256(inventory: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    for row in inventory["files"]:
        digest.update(str(row["path"]).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(row["bytes"]).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(row["sha256"]).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def run_code_identity() -> dict[str, Any]:
    paths = [
        (name, RUN_DIR / name)
        for name in (
            "02_train.py",
            "03_verify.py",
            "run_config.py",
            "training.py",
            "verification.py",
        )
    ]
    paths.extend(
        (f"../../src/sparsity_research/{name}", REPO_ROOT / "src" / "sparsity_research" / name)
        for name in (
            "artifacts.py",
            "capture.py",
            "data.py",
            "evaluation.py",
            "metrics.py",
            "optimization.py",
            "pressure.py",
            "pythia.py",
            "sites.py",
        )
    )
    files = []
    for name, path in paths:
        payload = path.read_bytes()
        files.append(
            {
                "path": name,
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    digest = hashlib.sha256()
    for row in files:
        digest.update(row["path"].encode("utf-8"))
        digest.update(b"\0")
        digest.update(row["sha256"].encode("ascii"))
        digest.update(b"\n")
    return {"files": files, "content_sha256": digest.hexdigest()}


def require_cuda(torch: Any) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; the active Python environment is CPU-only.")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("The approved BF16 run requires CUDA BF16 support.")


def seed_everything(torch: Any, seed: int) -> None:
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))


def require_training_cache(metadata: Mapping[str, Any]) -> None:
    expected = {
        "documents": TRAIN_CACHE_DOCUMENTS,
        "tokens": TRAIN_CACHE_TOKENS,
        "tokens_sha256": TRAIN_CACHE_SHA256,
        "block_size": 2048,
        "split": "train",
    }
    mismatches = [key for key, value in expected.items() if metadata.get(key) != value]
    complete, tail = divmod(int(metadata.get("tokens", -1)), 2048)
    if complete != 728_374:
        mismatches.append("complete_blocks")
    if tail != 1_464:
        mismatches.append("excluded_tail_tokens")
    if mismatches:
        raise ValueError("Training MiniPile cache mismatch: " + ", ".join(mismatches))


def require_validation_coverage(
    result: Mapping[str, Any], config: Mapping[str, Any]
) -> None:
    validation = mapping(config, "validation")
    expected = {
        "sequences": int(validation["complete_sequences"]),
        "input_tokens": int(validation["input_tokens"]),
        "excluded_tail_tokens": int(validation["excluded_tail_tokens"]),
        "complete_block_coverage": True,
    }
    mismatches = [key for key, value in expected.items() if result.get(key) != value]
    if mismatches:
        raise RuntimeError("Validation coverage mismatch: " + ", ".join(mismatches))


def cache_identity(metadata: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "dataset_name",
        "dataset_revision",
        "split",
        "text_column",
        "tokenizer_name",
        "tokenizer_revision",
        "documents",
        "tokens",
        "tokens_bytes",
        "tokens_sha256",
        "block_size",
        "append_eos",
        "complete_blocks",
        "evaluated_complete_block_tokens",
        "excluded_tail_tokens",
    )
    return {key: metadata.get(key) for key in keys}


def baseline_identity(config: Mapping[str, Any]) -> dict[str, Any]:
    comparison = mapping(config, "comparison")
    path = repo_path(str(comparison["baseline_verification"]))
    return {
        "path": path.relative_to(REPO_ROOT).as_posix(),
        "sha256": file_sha256(path),
        "gelu_attempt_id": comparison["gelu_attempt_id"],
        "relu_attempt_id": comparison["relu_attempt_id"],
        "gradient_clip_norm": float(comparison["baseline_gradient_clip_norm"]),
    }


def mapping(config: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = config.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"{key} must be a mapping.")
    return value


def repo_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def percentile(values: Any, probability: float) -> float:
    samples = sorted(float(value) for value in values)
    if not samples or any(
        not math.isfinite(value) or value <= 0.0 for value in samples
    ):
        raise ValueError("Timing samples must be positive and finite.")
    return samples[max(0, math.ceil(probability * len(samples)) - 1)]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)

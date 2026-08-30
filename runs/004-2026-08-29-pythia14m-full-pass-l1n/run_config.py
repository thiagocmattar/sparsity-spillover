"""Run 004 condition matrix, recipe invariants, cache checks, and identities."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
from typing import Any, Mapping

import yaml

from sparsity_research.data import (
    build_training_schedule,
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
COMPLETE_TRAIN_BLOCKS = 728_374
TRAIN_TAIL_TOKENS = 1_464
EXPECTED_WEIGHTS = (0.05, 0.1, 0.5, 1.0)
EXPECTED_CONDITION_IDS = (
    "gelu-control",
    "relu-control",
    "relu-l1n-0p05",
    "relu-l1n-0p1",
    "relu-l1n-0p5",
    "relu-l1n-1",
)
EXPECTED_WORKERS = {
    "controls": ("gelu-control", "relu-control"),
    "relu-l1n-0p05": ("relu-l1n-0p05",),
    "relu-l1n-0p1": ("relu-l1n-0p1",),
    "relu-l1n-0p5": ("relu-l1n-0p5",),
    "relu-l1n-1": ("relu-l1n-1",),
}
EXPECTED_MODEL_CHECKPOINTS = (0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 712)
EXPECTED_OPTIMIZER_CHECKPOINTS = (256, 512, 712)


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Run config must be a mapping.")
    validate_config(config)
    return config


def validate_config(config: Mapping[str, Any]) -> None:
    model = mapping(config, "model")
    expected_model = {
        "initialization": "random",
        "initialization_method": "small_init",
        "output_layer_initialization_method": "wang_init",
        "attention_implementation": "sdpa_flash",
        "execution_engine": "transformers_recipe_mapping",
    }
    if any(model.get(key) != value for key, value in expected_model.items()):
        raise ValueError("Model construction must match the approved Pythia recipe mapping.")

    recipe = mapping(config, "recipe")
    if recipe.get("exact_framework_reproduction") is not False:
        raise ValueError("Run 004 must disclose that the framework mapping is not bitwise NeoX.")
    if (
        recipe.get("optimizer_mapping") != "pytorch_fused_adamw"
        or tuple(recipe.get("weight_decay_exclusions", ())) != ("bias", "layer_norm")
        or recipe.get("lr_schedule_semantics") != "gpt_neox_v1_pre_step"
    ):
        raise ValueError("The approved GPT-NeoX optimizer/scheduler mapping changed.")
    runtime = mapping(config, "runtime")
    if dict(runtime) != {
        "python": "3.12",
        "torch": "2.11.0",
        "transformers": "5.12.1",
        "cuda_runtime": "12.8",
    }:
        raise ValueError("The approved RunPod software mapping changed.")

    conditions = mapping(config, "conditions")
    if conditions.get("pressure_method") != "l1_naive" or conditions.get("pressure_site") != "h":
        raise ValueError("Run 004 uses only naive L1 pressure at h.")
    weights = tuple(float(value) for value in conditions.get("relu_pressure_weights", ()))
    if weights != EXPECTED_WEIGHTS:
        raise ValueError("The approved ReLU L1N weights are 0.05, 0.1, 0.5, and 1.0.")
    if conditions.get("include_gelu_control") is not True or conditions.get("include_relu_control") is not True:
        raise ValueError("Both approved controls are required.")
    if tuple(row["id"] for row in condition_specs(config)) != EXPECTED_CONDITION_IDS:
        raise ValueError("The six-condition identity/order changed.")

    data = mapping(config, "data")
    if int(data.get("sequence_length", 0)) != 2048:
        raise ValueError("Pythia recipe sequence length must be 2,048.")
    seeds = mapping(config, "seeds")
    if int(seeds.get("model", -1)) != 1234 or int(seeds.get("data_order", -1)) != 1234:
        raise ValueError("Undeclared seeds must match the Pythia default 1234.")

    training = mapping(config, "training")
    integers = ("max_steps", "global_batch_size", "micro_batch_size", "gradient_accumulation_steps")
    for key in integers:
        value = training.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"training.{key} must be a positive integer.")
    expected_training = {
        "max_steps": 712,
        "global_batch_size": 1024,
        "micro_batch_size": 32,
        "gradient_accumulation_steps": 32,
        "optimizer": "adamw",
        "precision": "float16_dynamic",
        "parameter_dtype": "float32",
        "device": "cuda",
        "activation_checkpointing": False,
    }
    if any(training.get(key) != value for key, value in expected_training.items()):
        raise ValueError("Training decomposition or recipe mapping changed.")
    if training["micro_batch_size"] * training["gradient_accumulation_steps"] != training["global_batch_size"]:
        raise ValueError("Microbatch times accumulation must equal global batch.")
    float_expected = {
        "peak_learning_rate": 1e-3,
        "minimum_learning_rate": 1e-4,
        "gradient_clip_norm": 1.0,
        "weight_decay": 0.1,
        "warmup_fraction": 0.01,
        "hidden_dropout": 0.0,
        "attention_dropout": 0.0,
        "minimum_loss_scale": 1.0,
    }
    if any(float(training.get(key, math.nan)) != value for key, value in float_expected.items()):
        raise ValueError("A locked Pythia training scalar changed.")
    if tuple(float(value) for value in training.get("adamw_betas", ())) != (0.9, 0.95):
        raise ValueError("AdamW betas must match Pythia.")
    if float(training.get("adamw_eps", math.nan)) != 1e-8:
        raise ValueError("AdamW epsilon must match Pythia.")
    if (
        int(training.get("initial_loss_scale_power", -1)) != 12
        or int(training.get("loss_scale_window", -1)) != 1000
        or int(training.get("loss_scale_hysteresis", -1)) != 2
    ):
        raise ValueError("FP16 dynamic scaling must match the declared Pythia settings.")

    validation = mapping(config, "validation")
    expected_validation = {
        "documents": 500,
        "complete_sequences": 338,
        "input_tokens": 692_224,
        "excluded_tail_tokens": 1_444,
        "batch_size": 32,
    }
    if any(int(validation.get(key, -1)) != value for key, value in expected_validation.items()):
        raise ValueError("Complete MiniPile validation coverage changed.")

    diagnostics = mapping(config, "diagnostics")
    if tuple(diagnostics.get("activation_sites", ())) != (
        "m", "h", "q_post", "k_post", "v", "attention_output"
    ):
        raise ValueError("The approved terminal activation sites changed.")
    if tuple(float(value) for value in diagnostics.get("near_zero_thresholds", ())) != (0.0, 0.001, 0.01):
        raise ValueError("The approved near-zero thresholds changed.")
    if diagnostics.get("logical_products") is not True or int(diagnostics.get("logical_product_batch_size", 0)) != 1:
        raise ValueError("Logical-product capture must use the declared bounded batch size.")

    checkpoints = mapping(config, "checkpoints")
    if tuple(int(value) for value in checkpoints.get("model_steps", ())) != EXPECTED_MODEL_CHECKPOINTS:
        raise ValueError("Pythia learning-dynamics checkpoint cadence changed.")
    if tuple(int(value) for value in checkpoints.get("optimizer_steps", ())) != EXPECTED_OPTIMIZER_CHECKPOINTS:
        raise ValueError("Recovery checkpoint cadence changed.")

    runpod = mapping(config, "runpod")
    assignments = {
        str(key): tuple(str(item) for item in value)
        for key, value in mapping(runpod, "worker_assignments").items()
    }
    if assignments != EXPECTED_WORKERS or int(runpod.get("pod_count", 0)) != 5:
        raise ValueError("Five-Pod worker assignment changed.")
    if runpod.get("image") != (
        "runpod/pytorch@sha256:"
        "0a360022e8de4375af99430f84e8b38951acc397252163a37ceac7204d01be35"
    ):
        raise ValueError("The pinned RunPod base-image digest changed.")
    expected_runpod_integers = {
        "container_disk_gb": 30,
        "pod_volume_gb": 25,
        "terminate_after_hours": 7,
        "preflight_max_minutes": 45,
        "monitoring_interval_minutes": 5,
        "stale_event_minutes": 10,
    }
    if any(int(runpod.get(key, -1)) != value for key, value in expected_runpod_integers.items()):
        raise ValueError("The approved RunPod storage/guard/monitoring definition changed.")
    if runpod.get("secure_fallback") is not False:
        raise ValueError("Secure fallback requires separate approval.")


def condition_specs(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    weights = [float(value) for value in mapping(config, "conditions")["relu_pressure_weights"]]
    rows = [
        _condition(1, "gelu", None),
        _condition(2, "relu", None),
    ]
    rows.extend(_condition(index, "relu", weight) for index, weight in enumerate(weights, 3))
    return rows


def _condition(order: int, activation: str, weight: float | None) -> dict[str, Any]:
    control = weight is None
    label = "control" if control else _weight_label(float(weight))
    return {
        "id": f"{activation}-{label}",
        "order": int(order),
        "activation": activation,
        "pressure_method": "none" if control else "l1_naive",
        "pressure_sites": [] if control else ["h"],
        "pressure_weight": 0.0 if control else float(weight),
        "label": "control" if control else f"lambda={float(weight):g}",
        "is_control": control,
    }


def worker_conditions(config: Mapping[str, Any], worker_id: str) -> list[dict[str, Any]]:
    assignments = mapping(mapping(config, "runpod"), "worker_assignments")
    if worker_id not in assignments:
        raise KeyError(f"Unknown worker {worker_id!r}.")
    by_id = {row["id"]: row for row in condition_specs(config)}
    return [deepcopy(by_id[str(condition_id)]) for condition_id in assignments[worker_id]]


def resolved_condition_config(config: Mapping[str, Any], condition: Mapping[str, Any]) -> dict[str, Any]:
    resolved = deepcopy(dict(config))
    resolved["condition"] = deepcopy(dict(condition))
    if condition["activation"] == "gelu":
        topology_id, site_gate = "A0", None
    else:
        topology_id, site_gate = "A1-H", {"operator": "relu"}
    resolve_topology_and_gate(topology_id, site_gate)
    resolved["model"]["topology_id"] = topology_id
    resolved["model"]["site_gate"] = site_gate
    pressure = {
        "method": condition["pressure_method"],
        "sites": list(condition["pressure_sites"]),
        "weight": float(condition["pressure_weight"]),
        "step_budget": None,
        "eps": 1e-12,
    }
    parse_pressure_config(pressure)
    resolved["activation_pressure"] = pressure
    return resolved


def load_verified_caches(config: Mapping[str, Any], *, np: Any) -> tuple[Any, Any, dict[str, Any], dict[str, Any], float]:
    from time import perf_counter

    started = perf_counter()
    data = mapping(config, "data")
    train_metadata_path = repo_path(str(data["training_metadata"]))
    validation_metadata_path = repo_path(str(data["validation_metadata"]))
    train_metadata = json.loads(train_metadata_path.read_text(encoding="utf-8"))
    validation_metadata = json.loads(validation_metadata_path.read_text(encoding="utf-8"))
    train_path = verify_token_cache(train_metadata, base=train_metadata_path.parent)
    validation_path = verify_token_cache(validation_metadata, base=validation_metadata_path.parent)
    require_training_cache(train_metadata)
    require_full_minipile_validation(validation_metadata)
    train_tokens = np.memmap(train_path, dtype=np.int32, mode="r", shape=(int(train_metadata["tokens"]),))
    validation_tokens = np.memmap(validation_path, dtype=np.int32, mode="r", shape=(int(validation_metadata["tokens"]),))
    return train_tokens, validation_tokens, train_metadata, validation_metadata, perf_counter() - started


def build_schedule(config: Mapping[str, Any], train_metadata: Mapping[str, Any], *, np: Any):
    training = mapping(config, "training")
    result = build_training_schedule(
        np,
        token_count=int(train_metadata["tokens"]),
        block_size=int(mapping(config, "data")["sequence_length"]),
        max_steps=int(training["max_steps"]),
        gradient_accumulation_steps=int(training["gradient_accumulation_steps"]),
        micro_batch_size=int(training["micro_batch_size"]),
        seed=int(mapping(config, "seeds")["data_order"]),
    )
    metadata = result[2]
    if int(metadata["scheduled_blocks"]) != 729_088 or int(metadata["wrapped_blocks"]) != 714:
        raise RuntimeError("Full-pass schedule arithmetic changed.")
    return result


def microbatches_for_step(tokens: Any, step_starts: Any, *, block_size: int, device: Any, torch: Any, np: Any) -> list[Any]:
    return [
        torch.as_tensor(
            np.stack([tokens[int(start): int(start) + int(block_size)] for start in micro_starts]),
            dtype=torch.long,
            device=device,
        )
        for micro_starts in step_starts
    ]


def require_cuda(torch: Any) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the approved FP16 run.")
    if not bool(torch.backends.cuda.flash_sdp_enabled()):
        raise RuntimeError("The approved SDPA flash backend is disabled.")
    available = getattr(torch.backends.cuda, "is_flash_attention_available", None)
    if available is None or not bool(available()):
        raise RuntimeError("This PyTorch build has no CUDA flash-attention kernel.")


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
    if complete != COMPLETE_TRAIN_BLOCKS:
        mismatches.append("complete_blocks")
    if tail != TRAIN_TAIL_TOKENS:
        mismatches.append("excluded_tail_tokens")
    if mismatches:
        raise ValueError("Training MiniPile cache mismatch: " + ", ".join(mismatches))


def require_validation_coverage(result: Mapping[str, Any], config: Mapping[str, Any]) -> None:
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


def cache_identity(metadata: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "dataset_name", "dataset_revision", "split", "text_column", "tokenizer_name",
        "tokenizer_revision", "documents", "tokens", "tokens_bytes", "tokens_sha256",
        "block_size", "append_eos", "complete_blocks", "evaluated_complete_block_tokens",
        "excluded_tail_tokens",
    )
    return {key: metadata.get(key) for key in keys}


def run_code_identity() -> dict[str, Any]:
    names = (
        "02_train.py", "03_verify.py", "04_monitor.py", "run_config.py",
        "initialization.py", "optimizer_boundary.py", "diagnostics.py",
        "training.py", "verification.py",
        "../../src/sparsity_research/artifacts.py",
        "../../src/sparsity_research/capture.py",
        "../../src/sparsity_research/ceilings.py",
        "../../src/sparsity_research/data.py",
        "../../src/sparsity_research/evaluation.py",
        "../../src/sparsity_research/logical_capture.py",
        "../../src/sparsity_research/metrics.py",
        "../../src/sparsity_research/optimization.py",
        "../../src/sparsity_research/pressure.py",
        "../../src/sparsity_research/pythia.py",
        "../../src/sparsity_research/sites.py",
    )
    files = []
    for name in names:
        path = RUN_DIR / name
        payload = path.read_bytes()
        files.append({"path": name, "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()})
    digest = hashlib.sha256()
    for row in files:
        digest.update(row["path"].encode("utf-8"))
        digest.update(b"\0")
        digest.update(row["sha256"].encode("ascii"))
        digest.update(b"\n")
    return {"files": files, "content_sha256": digest.hexdigest()}


def git_identity() -> dict[str, Any]:
    executable = shutil.which("git")
    if executable is None:
        raise RuntimeError("Git is unavailable; definitive attempt identity is required.")
    commit = subprocess.run([executable, "rev-parse", "HEAD"], cwd=REPO_ROOT, check=True, capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run([executable, "status", "--porcelain"], cwd=REPO_ROOT, check=True, capture_output=True, text=True).stdout.strip())
    if len(commit) != 40:
        raise RuntimeError("Git commit identity is malformed.")
    return {"git_commit": commit, "git_dirty": dirty}


def mapping(config: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = config.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"{key} must be a mapping.")
    return value


def repo_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _weight_label(value: float) -> str:
    return f"l1n-{value:g}".replace(".", "p")

"""Approved Run 006 identities, conditions, cache checks, and schedule helpers."""

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
MODEL_REVISION = "7386d9a4ae45aef494a6e704910394def3037fc5"
EXPECTED_THRESHOLDS = (0.0, 0.01, 0.05, 0.1, 0.5)
EXPECTED_ACTIVE_SITES = ("a", "m", "h", "z")
EXPECTED_SCHEDULE_SHA256 = "c254893f0ea521e5834405d7a4e6edaed74472733d533aff68fb119e600151d4"


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Run config must be a mapping.")
    validate_config(config)
    return config


def validate_config(config: Mapping[str, Any]) -> None:
    model = mapping(config, "model")
    if model.get("architecture") != "EleutherAI/pythia-14m-deduped":
        raise ValueError("Run 006 requires the pinned Pythia-14M architecture.")
    if model.get("revision") != MODEL_REVISION or model.get("initialization") != "random":
        raise ValueError("Run 006 requires the pinned random-initialization contract.")
    expected_gate = {"operator": "one_sided_threshold", "kappa": 0.0}
    if model.get("topology_id") != "A4-Z" or model.get("site_gate") != expected_gate:
        raise ValueError("The base Run 006 model must declare A4-Z at kappa=0.")
    topology, gate = resolve_topology_and_gate(str(model["topology_id"]), dict(model["site_gate"]))
    if topology.active_sites != EXPECTED_ACTIVE_SITES or gate != expected_gate:
        raise ValueError("A4-Z did not resolve to the approved sites and gate.")

    conditions = mapping(config, "conditions")
    if tuple(conditions.get("active_sites", ())) != EXPECTED_ACTIVE_SITES:
        raise ValueError("The approved joint threshold sites changed.")
    if conditions.get("gate_operator") != "one_sided_threshold":
        raise ValueError("Run 006 requires the one-sided threshold gate.")
    if tuple(float(value) for value in conditions.get("gate_thresholds", ())) != EXPECTED_THRESHOLDS:
        raise ValueError("The approved kappa grid changed.")
    if conditions.get("pressure_method") != "none":
        raise ValueError("Run 006 must not use L1 or OL1 pressure.")

    training = mapping(config, "training")
    for key in ("max_steps", "global_batch_size", "micro_batch_size", "gradient_accumulation_steps"):
        value = training.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"training.{key} must be a positive integer.")
    if int(training["max_steps"]) != 581:
        raise ValueError("Run 006 must retain Run 005's 581-step token budget.")
    if int(training["micro_batch_size"]) * int(training["gradient_accumulation_steps"]) != int(
        training["global_batch_size"]
    ):
        raise ValueError("Microbatch times accumulation must equal global batch.")
    if int(training["global_batch_size"]) != 64:
        raise ValueError("The matched global batch is 64 sequences.")
    if float(training.get("peak_learning_rate", -1.0)) != 0.001:
        raise ValueError("The approved peak learning rate is 1e-3.")
    if float(training.get("target_cohort_seconds", -1.0)) != 3600.0:
        raise ValueError("The local ETC ceiling is 3,600 seconds.")
    if float(training.get("planning_cohort_seconds", -1.0)) != 3540.0:
        raise ValueError("The prelaunch planning envelope is 3,540 seconds.")
    if float(training.get("gradient_clip_norm", -1.0)) != 1.0:
        raise ValueError("The task-gradient clip norm must be 1.0.")
    if training.get("optimizer") != "adamw":
        raise ValueError("Run 006 requires AdamW.")
    if training.get("device") != "cuda" or training.get("precision") != "bfloat16":
        raise ValueError("The approved local path requires CUDA BF16.")
    for key in ("adamw_eps", "weight_decay", "warmup_fraction", "minimum_learning_rate_ratio"):
        value = float(training.get(key, math.nan))
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"training.{key} must be positive and finite.")

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
    if any(int(validation.get(key, -1)) != value for key, value in expected_validation.items()):
        raise ValueError("Complete MiniPile validation coverage changed.")

    diagnostics = mapping(config, "diagnostics")
    expected_diagnostics = (
        "a", "m", "h", "q_post", "k_post", "v", "z", "attention_output"
    )
    if tuple(diagnostics.get("activation_sites", ())) != expected_diagnostics:
        raise ValueError("The approved terminal activation sites changed.")
    if tuple(float(value) for value in diagnostics.get("near_zero_thresholds", ())) != (
        0.0, 0.001, 0.01
    ):
        raise ValueError("The approved near-zero thresholds changed.")
    if diagnostics.get("gradient_interaction") is not False:
        raise ValueError("Gradient-interaction metrics are inapplicable without pressure.")
    if diagnostics.get("logical_products") is not True:
        raise ValueError("Logical-product counters were approved.")
    if int(diagnostics.get("logical_product_batch_size", 0)) != 1:
        raise ValueError("Logical-product capture must use bounded batch size 1.")
    if diagnostics.get("clipping_frontier") is not None:
        raise ValueError("A clipping frontier was not approved.")
    artifacts = mapping(config, "artifacts")
    if artifacts.get("save_final_checkpoint") is not True or artifacts.get("save_optimizer") is not False:
        raise ValueError("Every final model checkpoint, without optimizer state, must be retained.")
    if len(condition_specs(config)) != 5:
        raise ValueError("Exactly five matched conditions are required.")


def condition_specs(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    conditions = mapping(config, "conditions")
    specs = []
    for order, threshold in enumerate(conditions["gate_thresholds"], start=1):
        value = float(threshold)
        specs.append(
            {
                "id": f"a4z-one-sided-kappa-{_threshold_label(value)}",
                "order": order,
                "topology_id": "A4-Z",
                "active_sites": list(EXPECTED_ACTIVE_SITES),
                "gate_operator": "one_sided_threshold",
                "gate_threshold": value,
                "pressure_method": "none",
                "pressure_sites": [],
                "pressure_weight": 0.0,
                "step_budget": None,
                "label": f"kappa={value:g}",
                "is_control": value == 0.0,
            }
        )
    return specs


def resolved_condition_config(
    config: Mapping[str, Any], condition: Mapping[str, Any]
) -> dict[str, Any]:
    resolved = deepcopy(dict(config))
    resolved["condition"] = deepcopy(dict(condition))
    resolved["model"]["site_gate"] = {
        "operator": "one_sided_threshold",
        "kappa": float(condition["gate_threshold"]),
    }
    pressure = {
        "method": "none",
        "sites": [],
        "weight": 0.0,
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
    validation_metadata = json.loads(validation_metadata_path.read_text(encoding="utf-8"))
    train_path = verify_token_cache(train_metadata, base=train_metadata_path.parent)
    validation_path = verify_token_cache(validation_metadata, base=validation_metadata_path.parent)
    require_training_cache(train_metadata)
    require_full_minipile_validation(validation_metadata)
    train_tokens = np.memmap(train_path, dtype=np.int32, mode="r", shape=(int(train_metadata["tokens"]),))
    validation_tokens = np.memmap(
        validation_path, dtype=np.int32, mode="r", shape=(int(validation_metadata["tokens"]),)
    )
    return train_tokens, validation_tokens, train_metadata, validation_metadata, perf_counter() - started


def build_schedule(
    config: Mapping[str, Any], train_metadata: Mapping[str, Any], *, np: Any
) -> tuple[Any, str, dict[str, Any]]:
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
    if result[1] != EXPECTED_SCHEDULE_SHA256:
        raise RuntimeError("Run 006 no longer matches Run 005's realized training schedule.")
    return result


def microbatches_for_step(
    tokens: Any, step_starts: Any, *, block_size: int, device: Any, torch: Any, np: Any
) -> list[Any]:
    return [
        torch.as_tensor(
            np.stack([tokens[int(start): int(start) + int(block_size)] for start in micro_starts]),
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
    local_names = (
        "01_calibrate.py", "02_train.py", "03_verify.py", "04_launch.ps1",
        "05_monitor.ps1", "calibration.py", "diagnostics.py", "launch_detached.py",
        "run_config.py", "training.py", "verification.py",
    )
    shared_names = (
        "artifacts.py", "capture.py", "ceilings.py", "data.py", "evaluation.py",
        "logical_capture.py", "metrics.py", "optimization.py", "pressure.py",
        "pythia.py", "sites.py",
    )
    paths = [(name, RUN_DIR / name) for name in local_names]
    paths.extend(
        (f"../../src/sparsity_research/{name}", REPO_ROOT / "src" / "sparsity_research" / name)
        for name in shared_names
    )
    files = []
    for name, path in paths:
        payload = path.read_bytes()
        files.append({"path": name, "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()})
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
        "documents": TRAIN_CACHE_DOCUMENTS, "tokens": TRAIN_CACHE_TOKENS,
        "tokens_sha256": TRAIN_CACHE_SHA256, "block_size": 2048, "split": "train",
    }
    mismatches = [key for key, value in expected.items() if metadata.get(key) != value]
    complete, tail = divmod(int(metadata.get("tokens", -1)), 2048)
    if complete != 728_374:
        mismatches.append("complete_blocks")
    if tail != 1_464:
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


def cache_identity(metadata: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "dataset_name", "dataset_revision", "split", "text_column", "tokenizer_name",
        "tokenizer_revision", "documents", "tokens", "tokens_bytes", "tokens_sha256",
        "block_size", "append_eos", "complete_blocks", "evaluated_complete_block_tokens",
        "excluded_tail_tokens",
    )
    return {key: metadata.get(key) for key in keys}


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
    if not samples or any(not math.isfinite(value) or value <= 0.0 for value in samples):
        raise ValueError("Timing samples must be positive and finite.")
    return samples[max(0, math.ceil(probability * len(samples)) - 1)]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _threshold_label(value: float) -> str:
    return f"{value:g}".replace(".", "p")

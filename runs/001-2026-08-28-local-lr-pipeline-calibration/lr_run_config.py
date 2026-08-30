"""Approved condition contract, cache identity, and deterministic schedule helpers."""

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


RUN_DIR = Path(__file__).resolve().parent
REPO_ROOT = RUN_DIR.parents[1]
DEFAULT_CONFIG = RUN_DIR / "config.yaml"
TRAIN_CACHE_SHA256 = "da82a2ea2e0080c7fd681c7a93b07d3d9ff3d5357a8640895a82d536a1eaf97c"
TRAIN_CACHE_TOKENS = 1_491_711_416
TRAIN_CACHE_DOCUMENTS = 1_000_000


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Run config must be a mapping.")
    validate_config(config)
    return config


def validate_config(config: Mapping[str, Any]) -> None:
    model = mapping(config, "model")
    if model.get("initialization") != "random":
        raise ValueError("This pretraining run requires random initialization.")
    if model.get("topology_id") != "A0" or model.get("site_gate") is not None:
        raise ValueError("This LR control requires topology A0 with no gate.")

    raw_rates = mapping(config, "conditions").get("peak_learning_rates")
    if not isinstance(raw_rates, list) or len(raw_rates) != 4:
        raise ValueError("Exactly four peak learning rates are required.")
    rates = [float(value) for value in raw_rates]
    if any(not math.isfinite(value) or value <= 0.0 for value in rates):
        raise ValueError("Peak learning rates must be positive and finite.")
    if rates != sorted(set(rates)):
        raise ValueError("Peak learning rates must be unique and increasing.")

    training = mapping(config, "training")
    for key in (
        "max_steps",
        "global_batch_size",
        "micro_batch_size",
        "gradient_accumulation_steps",
    ):
        if isinstance(training.get(key), bool) or int(training.get(key, 0)) <= 0:
            raise ValueError(f"training.{key} must be a positive integer.")
    realized_global = int(training["micro_batch_size"]) * int(
        training["gradient_accumulation_steps"]
    )
    if realized_global != int(training["global_batch_size"]):
        raise ValueError("Microbatch times accumulation must equal global batch size.")
    if training.get("device") != "cuda" or training.get("precision") != "bfloat16":
        raise ValueError("The approved local design requires CUDA BF16.")
    if float(training.get("gradient_clip_norm", -1.0)) != 1.0:
        raise ValueError("The operational contract requires gradient clipping at 1.0.")

    data = mapping(config, "data")
    if int(data.get("sequence_length", 0)) != 2048:
        raise ValueError("The approved data contract requires 2,048-token sequences.")
    validation = mapping(config, "validation")
    expected_validation = {
        "documents": 500,
        "complete_sequences": 338,
        "input_tokens": 692_224,
        "excluded_tail_tokens": 1_444,
    }
    if any(int(validation.get(key, -1)) != value for key, value in expected_validation.items()):
        raise ValueError("Complete MiniPile validation coverage fields do not match the contract.")
    parse_pressure_config(dict(mapping(config, "activation_pressure")))


def resolved_condition_config(config: Mapping[str, Any], peak_lr: float) -> dict[str, Any]:
    resolved = deepcopy(dict(config))
    resolved["condition"] = {
        "id": f"lr-{learning_rate_label(peak_lr)}",
        "peak_learning_rate": float(peak_lr),
    }
    resolved["training"]["peak_learning_rate"] = float(peak_lr)
    return resolved


def learning_rate_label(value: float) -> str:
    return f"{float(value):.0e}".replace("e-0", "e-").replace("e+0", "e+")


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
    validation_path = verify_token_cache(
        validation_metadata, base=validation_metadata_path.parent
    )
    require_training_cache(train_metadata)
    require_full_minipile_validation(validation_metadata)
    train_tokens = np.memmap(
        train_path, dtype=np.int32, mode="r", shape=(int(train_metadata["tokens"]),)
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


def build_schedule(config: Mapping[str, Any], train_metadata: Mapping[str, Any], *, np: Any):
    training = mapping(config, "training")
    seeds = mapping(config, "seeds")
    return build_training_schedule(
        np,
        token_count=int(train_metadata["tokens"]),
        block_size=int(mapping(config, "data")["sequence_length"]),
        max_steps=int(training["max_steps"]),
        gradient_accumulation_steps=int(training["gradient_accumulation_steps"]),
        micro_batch_size=int(training["micro_batch_size"]),
        seed=int(seeds["data_order"]),
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
    batches = []
    for micro_starts in step_starts:
        array = np.stack(
            [tokens[int(start) : int(start) + int(block_size)] for start in micro_starts]
        )
        batches.append(torch.as_tensor(array, dtype=torch.long, device=device))
    return batches


def parameter_sha256(model: Any) -> str:
    digest = hashlib.sha256()
    state = model.state_dict()
    for name in sorted(state):
        tensor = state[name].detach().contiguous().cpu()
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
    names = (
        "01_calibrate.py",
        "02_train.py",
        "lr_calibration.py",
        "lr_run_config.py",
        "lr_training.py",
        "pipeline.py",
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
        raise ValueError("Training MiniPile cache identity mismatch: " + ", ".join(mismatches))


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
    index = max(0, math.ceil(probability * len(samples)) - 1)
    return samples[index]


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

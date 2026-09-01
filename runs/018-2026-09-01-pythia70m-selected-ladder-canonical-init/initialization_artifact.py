"""Generate and load Run 018's hash-pinned random-pretraining initialization."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import random
from typing import Any, Mapping

import yaml

from initialization import apply_pythia_70m_initialization
from model_factory import build_pinned_run018_model
from run_config import (
    DEFAULT_CONFIG,
    EXPECTED_INITIAL_PARAMETER_SHA256,
    RUN_DIR,
    condition_specs,
    mapping,
    parameter_sha256,
    resolved_condition_config,
    seed_everything,
    validate_science_config,
)


ARTIFACT_DIR = RUN_DIR / "prelaunch" / "initialization"
MODEL_FILENAME = "pythia70m-seed1234.safetensors"
RNG_FILENAME = "pythia70m-seed1234-rng.pt"
METADATA_FILENAME = "metadata.json"


def sha256_file(path: Path, *, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def state_key_sha256(state: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name]
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(",".join(str(int(value)) for value in tensor.shape).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _raw_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Run 018 config must be a mapping.")
    return value


def _paths(config: Mapping[str, Any]) -> tuple[Path, Path, Path]:
    artifact = mapping(config, "initialization_artifact")
    return tuple(RUN_DIR / str(artifact[key]) for key in ("model_path", "rng_path", "metadata_path"))


def _build_generation_model(config: Mapping[str, Any], *, torch: Any, auto_model: Any) -> Any:
    condition = next(row for row in condition_specs(config) if row["id"] == "a0-gelu")
    resolved = resolved_condition_config(config, condition)
    seed = int(mapping(config, "seeds")["model"])
    seed_everything(torch, seed)
    model = build_pinned_run018_model(
        dict(mapping(resolved, "model")),
        device=torch.device("cpu"),
        torch=torch,
        auto_model=auto_model,
    )
    seed_everything(torch, seed)
    return model


def generate_initialization_artifacts(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    """Generate the approved draw once and prove a safetensors round trip."""

    import numpy as np
    import torch
    import transformers
    from safetensors.torch import load_file, save_file
    from transformers import AutoModelForCausalLM

    config = _raw_config(config_path)
    validate_science_config(config)
    model_path, rng_path, metadata_path = _paths(config)
    for path in (model_path, rng_path, metadata_path):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite canonical initialization artifact: {path}")
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    seed = int(mapping(config, "seeds")["model"])
    random.seed(seed)
    np.random.seed(seed)
    model = _build_generation_model(config, torch=torch, auto_model=AutoModelForCausalLM)
    initialization = apply_pythia_70m_initialization(model, torch=torch)
    realized = parameter_sha256(model)
    if realized != EXPECTED_INITIAL_PARAMETER_SHA256:
        raise RuntimeError(
            "Canonical generator did not reproduce the approved Run 017 draw: "
            f"realized={realized}, expected={EXPECTED_INITIAL_PARAMETER_SHA256}"
        )
    state = {name: tensor.detach().cpu().contiguous() for name, tensor in model.state_dict().items()}
    if len(state) != 76 or sum(t.numel() * t.element_size() for t in state.values()) != 281_706_496:
        raise RuntimeError("Canonical state tensor count or bytes changed.")
    key_hash = state_key_sha256(state)

    model_tmp = model_path.with_suffix(model_path.suffix + ".tmp")
    rng_tmp = rng_path.with_suffix(rng_path.suffix + ".tmp")
    save_file(
        state,
        str(model_tmp),
        metadata={
            "kind": "locally_generated_random_pretraining_initialization",
            "parameter_sha256": realized,
            "released_weights_loaded": "false",
        },
    )
    torch.save(
        {
            "schema_version": 1,
            "model_seed": seed,
            "parameter_sha256": realized,
            "python_rng_state": random.getstate(),
            "numpy_rng_state": np.random.get_state(),
            "torch_cpu_rng_state": torch.get_rng_state(),
        },
        rng_tmp,
    )
    model_tmp.replace(model_path)
    rng_tmp.replace(rng_path)

    del state, model
    roundtrip_model = _build_generation_model(config, torch=torch, auto_model=AutoModelForCausalLM)
    loaded = load_file(str(model_path), device="cpu")
    if state_key_sha256(loaded) != key_hash:
        raise RuntimeError("Canonical safetensors key identity changed on round trip.")
    roundtrip_model.load_state_dict(loaded, strict=True)
    roundtrip_hash = parameter_sha256(roundtrip_model)
    if roundtrip_hash != realized:
        raise RuntimeError("Canonical safetensors round trip changed parameter bytes.")

    metadata = {
        "schema_version": 1,
        "kind": "locally_generated_random_pretraining_initialization",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "released_weights_loaded": False,
        "source_run": "017-2026-09-01-pythia70m-selected-ladder-portable-init",
        "architecture": dict(mapping(config, "model")),
        "model_seed": int(mapping(config, "seeds")["model"]),
        "parameter_sha256": realized,
        "roundtrip_parameter_sha256": roundtrip_hash,
        "tensor_count": len(loaded),
        "tensor_bytes": sum(t.numel() * t.element_size() for t in loaded.values()),
        "state_key_sha256": key_hash,
        "model_artifact": {
            "path": str(model_path.relative_to(RUN_DIR)).replace("\\", "/"),
            "bytes": model_path.stat().st_size,
            "sha256": sha256_file(model_path),
            "format": "safetensors",
        },
        "rng_artifact": {
            "path": str(rng_path.relative_to(RUN_DIR)).replace("\\", "/"),
            "bytes": rng_path.stat().st_size,
            "sha256": sha256_file(rng_path),
            "restores": ["python", "numpy", "torch_cpu"],
            "cuda_policy": "seed_1234_on_worker_before_artifact_load",
        },
        "initialization_recipe": initialization,
        "generation_runtime": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metadata


def _verify_file(path: Path, *, expected_bytes: int, expected_sha256: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Required canonical initialization artifact is absent: {path}")
    if path.stat().st_size != int(expected_bytes):
        raise RuntimeError(f"Canonical artifact byte count changed: {path}")
    realized = sha256_file(path)
    if realized != expected_sha256:
        raise RuntimeError(
            f"Canonical artifact SHA-256 changed for {path.name}: "
            f"realized={realized}, expected={expected_sha256}"
        )


def load_pinned_initialization(model: Any, *, torch: Any) -> dict[str, Any]:
    """Load the approved local draw and restore its post-initialization CPU RNG state."""

    import numpy as np
    from safetensors.torch import load_file

    from run_config import load_config

    config = load_config()
    artifact = mapping(config, "initialization_artifact")
    model_path, rng_path, metadata_path = _paths(config)
    _verify_file(
        model_path,
        expected_bytes=int(artifact["model_bytes"]),
        expected_sha256=str(artifact["model_sha256"]),
    )
    _verify_file(
        rng_path,
        expected_bytes=int(artifact["rng_bytes"]),
        expected_sha256=str(artifact["rng_sha256"]),
    )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if (
        metadata.get("kind") != "locally_generated_random_pretraining_initialization"
        or metadata.get("released_weights_loaded") is not False
        or metadata.get("parameter_sha256") != EXPECTED_INITIAL_PARAMETER_SHA256
    ):
        raise RuntimeError("Canonical initialization metadata changed.")
    if {parameter.device.type for parameter in model.parameters()} != {"cpu"}:
        raise RuntimeError("Canonical initialization must load into a CPU model.")

    state = load_file(str(model_path), device="cpu")
    if len(state) != int(artifact["tensor_count"]):
        raise RuntimeError("Canonical initialization tensor count changed.")
    if sum(t.numel() * t.element_size() for t in state.values()) != int(artifact["tensor_bytes"]):
        raise RuntimeError("Canonical initialization tensor bytes changed.")
    if state_key_sha256(state) != metadata["state_key_sha256"]:
        raise RuntimeError("Canonical initialization state-key identity changed.")
    model.load_state_dict(state, strict=True)
    realized = parameter_sha256(model)
    if realized != EXPECTED_INITIAL_PARAMETER_SHA256:
        raise RuntimeError(
            "Canonical initialization changed after strict load: "
            f"realized={realized}, expected={EXPECTED_INITIAL_PARAMETER_SHA256}"
        )

    rng = torch.load(rng_path, map_location="cpu", weights_only=False)
    if (
        int(rng.get("schema_version", 0)) != 1
        or int(rng.get("model_seed", -1)) != int(mapping(config, "seeds")["model"])
        or rng.get("parameter_sha256") != EXPECTED_INITIAL_PARAMETER_SHA256
    ):
        raise RuntimeError("Canonical post-initialization RNG artifact changed.")
    random.setstate(rng["python_rng_state"])
    np.random.set_state(rng["numpy_rng_state"])
    torch.set_rng_state(rng["torch_cpu_rng_state"])

    recipe = dict(metadata["initialization_recipe"])
    recipe.update(
        realization="hash_pinned_generated_artifact",
        artifact_sha256=str(artifact["model_sha256"]),
        rng_artifact_sha256=str(artifact["rng_sha256"]),
        parameter_sha256=realized,
        released_weights_loaded=False,
    )
    model.config._attn_implementation = "sdpa"
    model.config.use_cache = False
    model.config.pythia_recipe_initialization = recipe
    return recipe

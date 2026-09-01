"""Run 017 worker lifecycle with 70M initialization and condition-driven OL1 capture."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

from _reuse_run004 import load_run004_module
from initialization import apply_pythia_70m_initialization
from model_factory import build_pinned_run017_model
from run017_capture import ConditionPressureCapture
from run_config import EXPECTED_INITIAL_PARAMETER_SHA256, write_json


_BASE = load_run004_module("_run017_frozen_run004_training", "training.py")
_SOURCE_PARAMETER_SHA256 = _BASE.parameter_sha256


def _transfer_initialized_model_to_cuda(model: Any) -> None:
    import torch

    model.to(device=torch.device("cuda"), dtype=torch.float32)
    if {parameter.device.type for parameter in model.parameters()} != {"cuda"}:
        raise RuntimeError("Run 017 model transfer to CUDA did not complete.")


def _verified_initial_parameter_sha256(model: Any) -> str:
    """Verify the CPU draw, then transfer it to CUDA before optimizer creation."""

    parameter_devices = {parameter.device.type for parameter in model.parameters()}
    if parameter_devices != {"cpu"}:
        raise RuntimeError(
            "Run 017 initialization must be realized entirely on CPU before hashing; "
            f"devices={sorted(parameter_devices)}"
        )

    realized = str(_SOURCE_PARAMETER_SHA256(model))
    if realized != EXPECTED_INITIAL_PARAMETER_SHA256:
        raise RuntimeError(
            "Pinned portable CPU initialization mismatch before training: "
            f"realized={realized}, expected={EXPECTED_INITIAL_PARAMETER_SHA256}"
        )
    _transfer_initialized_model_to_cuda(model)
    return realized


def _build_random_pythia(model_config: dict[str, Any], **kwargs: Any) -> Any:
    kwargs.pop("auto_config", None)
    torch = kwargs["torch"]
    kwargs["device"] = torch.device("cpu")
    model = build_pinned_run017_model(model_config, **kwargs)
    pressure_sites = tuple(str(site) for site in model_config.get("pressure_sites", ()))
    model.config.pressure_sites = list(pressure_sites)
    return model


def _microbatches_for_step(
    tokens: Any,
    step_starts: Any,
    *,
    block_size: int,
    device: Any,
    torch: Any,
    np: Any,
):
    """Defer cache slicing and host-to-device staging into the measured boundary."""

    return (
        torch.as_tensor(
            np.stack(
                [tokens[int(start) : int(start) + int(block_size)] for start in micro_starts]
            ),
            dtype=torch.long,
            device=device,
        )
        for micro_starts in step_starts
    )


def _save_checkpoint(
    *,
    model: Any,
    optimizer: Any,
    scaler: Any,
    step: int,
    include_optimizer: bool,
    root: Path,
    schedule_hash: str,
    torch: Any,
) -> Path:
    """Retain one complete final recovery checkpoint and no initial weight copy."""

    target = root / f"step_{int(step):06d}"
    if int(step) == 0:
        return target
    if target.exists():
        raise FileExistsError(f"Checkpoint already exists: {target}")
    if not include_optimizer:
        raise ValueError("Run 017 retains only a complete final recovery checkpoint.")
    import numpy as np

    target.mkdir(parents=True)
    model.save_pretrained(target, safe_serialization=True)
    state = {
        "step": int(step),
        "schedule_hash": schedule_hash,
        "loss_scaler": scaler.state_dict(),
        "optimizer_saved": True,
        "rng_saved": ["python", "numpy", "torch_cpu", "torch_cuda_all"],
    }
    torch.save(
        {
            **state,
            "optimizer": optimizer.state_dict(),
            "python_rng_state": random.getstate(),
            "numpy_rng_state": np.random.get_state(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_states": torch.cuda.get_rng_state_all(),
        },
        target / "training_state.pt",
    )
    write_json(target / "checkpoint_metadata.json", state)
    return target


_BASE.build_random_pythia = _build_random_pythia
_BASE.apply_pythia_14m_initialization = apply_pythia_70m_initialization
_BASE.ActivationCapture = ConditionPressureCapture
_BASE.microbatches_for_step = _microbatches_for_step
_BASE._save_checkpoint = _save_checkpoint
_BASE.parameter_sha256 = _verified_initial_parameter_sha256

run_worker = _BASE.run_worker
run_condition = _BASE.run_condition
timed_validation = _BASE.timed_validation

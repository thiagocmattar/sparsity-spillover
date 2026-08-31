"""Run 016 FP16 boundary for controls and dynamically verified A4/A7 OL1."""

from __future__ import annotations

import hashlib
from typing import Any, Iterable

from sparsity_research.optimization import prepare_adamw_gradients
from sparsity_research.pressure import (
    PressureConfig,
    accumulate_grads,
    activation_l1,
    apply_ol1_correction,
    clone_grads,
    gradient_interaction,
)
from sparsity_research.sites import resolve_topology

from _reuse_run004 import load_run004_module
from run_config import A4_SITES, A7_SITES


_BASE = load_run004_module("_run016_frozen_run004_optimizer_boundary", "optimizer_boundary.py")
DynamicLossScaler = _BASE.DynamicLossScaler
build_recipe_adamw = _BASE.build_recipe_adamw
recipe_learning_rate = _BASE.recipe_learning_rate
recipe_attention_context = _BASE.recipe_attention_context


def run_recipe_boundary(
    *,
    model: Any,
    optimizer: Any,
    batches: Iterable[Any],
    pressure: PressureConfig,
    capture: Any | None,
    loss_scaler: DynamicLossScaler,
    gradient_clip_norm: float,
    torch: Any,
    device: Any,
) -> dict[str, Any]:
    """Run a control boundary or one atomic task-AdamW/OL1 boundary."""

    if not pressure.enabled:
        if capture is not None:
            raise ValueError("Control conditions must not allocate a pressure capture.")
        return _BASE.run_recipe_boundary(
            model=model,
            optimizer=optimizer,
            batches=batches,
            pressure=pressure,
            capture=capture,
            loss_scaler=loss_scaler,
            gradient_clip_norm=gradient_clip_norm,
            torch=torch,
            device=device,
        )
    if not pressure.orthogonal:
        raise ValueError("Run 016 has no enabled pressure method other than orthogonal_l1.")
    expected_sites = _expected_pressure_sites(model, pressure)
    if capture is None:
        raise ValueError("OL1 requires exact-site activation capture.")

    microbatches = tuple(batches)
    if not microbatches:
        raise ValueError("At least one microbatch is required.")
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer.zero_grad(set_to_none=True)
    task_grads: list[Any | None] = []
    pressure_grads: list[Any | None] = []
    task_total = pressure_total = 0.0
    count = len(microbatches)
    scale = float(loss_scaler.scale)
    expected_names = _expected_pressure_capture_names(model, expected_sites)

    for batch in microbatches:
        capture.clear()
        with recipe_attention_context(torch, device), _BASE._autocast(torch, device):
            output = model(input_ids=batch, labels=batch)
            task_loss = output.loss
            captured_names = tuple(sorted(str(name) for name in capture.activations))
            if captured_names != expected_names:
                missing = sorted(set(expected_names) - set(captured_names))
                extra = sorted(set(captured_names) - set(expected_names))
                raise RuntimeError(
                    "Run 016 pressure capture mismatch: "
                    f"missing={missing}, extra={extra}"
                )
            pressure_loss = activation_l1(capture.activations)
        _BASE._finite_loss(task_loss, "task loss", torch)
        _BASE._finite_loss(pressure_loss, "pressure loss", torch)
        task_total += float(task_loss.detach().float().cpu())
        pressure_total += float(pressure_loss.detach().float().cpu())
        task_grads = accumulate_grads(
            task_grads,
            torch.autograd.grad(
                task_loss * (scale / count),
                parameters,
                retain_graph=True,
                allow_unused=True,
            ),
        )
        pressure_grads = accumulate_grads(
            pressure_grads,
            torch.autograd.grad(
                pressure_loss * (scale / count),
                parameters,
                allow_unused=True,
            ),
        )

    task_grads = _BASE._unscale(task_grads, scale)
    pressure_grads = _BASE._unscale(pressure_grads, scale)
    finite = _gradients_finite(task_grads, pressure_grads, torch=torch)
    scaler_metrics = loss_scaler.update(finite=finite)
    pressure_mean = pressure_total / count
    result: dict[str, Any] = {
        "task_loss": task_total / count,
        "pressure_loss": pressure_mean,
        "pressure_weight": pressure.weight,
        "weighted_pressure_loss": pressure.weight * pressure_mean,
        "augmented_loss": task_total / count + pressure.weight * pressure_mean,
        "optimizer_step_skipped": not finite,
        "gradient_overflow": not finite,
        "fp16_overflow_policy": "skip_entire_boundary",
        "pressure_sites": list(expected_sites),
        "pressure_capture_tensor_count": len(expected_names),
        "pressure_capture_names_sha256": _capture_names_sha256(expected_names),
        **scaler_metrics,
    }
    if not finite:
        optimizer.zero_grad(set_to_none=True)
        result.update(
            adamw_gradient_norm_pre_clip=None,
            adamw_gradient_norm_post_clip=None,
            adamw_gradient_clip_norm=float(gradient_clip_norm),
            adamw_gradient_clipping_enabled=True,
            adamw_gradient_was_clipped=None,
            task_gradient_norm=None,
            pressure_gradient_norm=None,
            pressure_to_task_gradient_norm_ratio=None,
            task_pressure_gradient_dot=None,
            task_pressure_gradient_cosine=None,
            gradient_conflict=None,
            ol1_correction_applied=False,
        )
        return result

    _assign_task_gradients(parameters, task_grads)
    clip = prepare_adamw_gradients(
        parameters,
        gradient_clip_norm=float(gradient_clip_norm),
        torch=torch,
    )
    clipped_task_grads = clone_grads(parameters)
    result.update(gradient_interaction(clipped_task_grads, pressure_grads, eps=pressure.eps))
    result.update(clip)
    optimizer.step()
    result.update(
        apply_ol1_correction(
            optimizer,
            parameters,
            clipped_task_grads,
            pressure_grads,
            pressure_weight=pressure.weight,
            step_budget=float(pressure.step_budget),
            eps=pressure.eps,
        )
    )
    result["ol1_correction_applied"] = True
    return result


def _expected_pressure_sites(model: Any, pressure: PressureConfig) -> tuple[str, ...]:
    topology = resolve_topology(getattr(model.config, "topology_id", None))
    expected = A4_SITES if topology.topology_id == "A4-Z" else A7_SITES if topology.topology_id == "A7-Z-POST" else ()
    configured = tuple(getattr(model.config, "pressure_sites", ()))
    observed = tuple(pressure.sites)
    if not expected or configured != expected or observed != expected:
        raise ValueError(
            "OL1 pressure sites must equal the resolved A4/A7 topology: "
            f"topology={topology.topology_id}, configured={configured}, pressure={observed}"
        )
    return expected


def _assign_task_gradients(parameters: list[Any], task_grads: list[Any | None]) -> None:
    if len(parameters) != len(task_grads):
        raise RuntimeError("Task-gradient list does not match trainable parameters.")
    for parameter, task_grad in zip(parameters, task_grads, strict=True):
        parameter.grad = None if task_grad is None else task_grad


def _gradients_finite(
    task_grads: list[Any | None], pressure_grads: list[Any | None], *, torch: Any
) -> bool:
    tensors = [gradient for gradient in (*task_grads, *pressure_grads) if gradient is not None]
    return bool(tensors) and all(bool(torch.isfinite(tensor).all().item()) for tensor in tensors)


def _expected_pressure_capture_names(model: Any, sites: tuple[str, ...]) -> tuple[str, ...]:
    layers = getattr(getattr(model, "gpt_neox", None), "layers", None)
    layer_count = len(layers) if layers is not None else int(getattr(model.config, "num_hidden_layers", 0))
    if layer_count <= 0:
        raise ValueError("Could not determine the model layer count for pressure capture.")
    return tuple(sorted(f"{site}.layer_{layer}" for site in sites for layer in range(layer_count)))


def _capture_names_sha256(names: tuple[str, ...]) -> str:
    return hashlib.sha256("\n".join(names).encode("utf-8")).hexdigest()

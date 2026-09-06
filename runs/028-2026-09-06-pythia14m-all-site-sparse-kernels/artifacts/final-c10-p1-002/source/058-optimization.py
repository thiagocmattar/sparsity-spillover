"""One accumulated AdamW boundary with none, naive L1, or OL1 pressure."""

from __future__ import annotations

from contextlib import nullcontext
import math
from typing import Any, Callable, Iterable

from .pressure import (
    PressureConfig,
    accumulate_grads,
    activation_l1,
    apply_ol1_correction,
    clone_grads,
    gradient_interaction,
)


GLOBAL_GRADIENT_CLIP_NORM = 1.0


def build_adamw(model: Any, config: dict[str, Any], *, torch: Any) -> Any:
    if config.get("optimizer") != "adamw":
        raise ValueError("Only AdamW is implemented.")
    betas = tuple(float(value) for value in config["adamw_betas"])
    if len(betas) != 2 or any(not 0.0 <= value < 1.0 for value in betas):
        raise ValueError("AdamW betas must contain two values in [0, 1).")
    return torch.optim.AdamW(
        model.parameters(),
        lr=float(config["peak_learning_rate"]),
        betas=betas,
        eps=float(config["adamw_eps"]),
        weight_decay=float(config["weight_decay"]),
    )


def run_optimizer_boundary(
    *,
    model: Any,
    optimizer: Any,
    batches: Iterable[Any],
    pressure: PressureConfig,
    capture: Any | None,
    torch: Any,
    device: Any,
    autocast_dtype: Any | None,
    gradient_clip_norm: float | None = GLOBAL_GRADIENT_CLIP_NORM,
    activation_observer: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Run one optimizer boundary over an explicit sequence of microbatches."""

    microbatches = tuple(batches)
    if not microbatches:
        raise ValueError("At least one microbatch is required.")
    if pressure.enabled and capture is None:
        raise ValueError("Pressure requires activation capture at its exact sites.")
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if pressure.orthogonal:
        return _ol1_boundary(
            model=model,
            optimizer=optimizer,
            parameters=parameters,
            microbatches=microbatches,
            pressure=pressure,
            capture=capture,
            torch=torch,
            device=device,
            autocast_dtype=autocast_dtype,
            gradient_clip_norm=gradient_clip_norm,
            observer=activation_observer,
        )
    return _standard_boundary(
        model=model,
        optimizer=optimizer,
        parameters=parameters,
        microbatches=microbatches,
        pressure=pressure,
        capture=capture,
        torch=torch,
        device=device,
        autocast_dtype=autocast_dtype,
        gradient_clip_norm=gradient_clip_norm,
        observer=activation_observer,
    )


def _standard_boundary(
    *,
    model: Any,
    optimizer: Any,
    parameters: list[Any],
    microbatches: tuple[Any, ...],
    pressure: PressureConfig,
    capture: Any | None,
    torch: Any,
    device: Any,
    autocast_dtype: Any | None,
    gradient_clip_norm: float | None,
    observer: Callable[[dict[str, Any]], None] | None,
) -> dict[str, Any]:
    optimizer.zero_grad(set_to_none=True)
    task_total = pressure_total = 0.0
    task_grads: list[Any | None] = []
    pressure_grads: list[Any | None] = []
    count = len(microbatches)
    for batch in microbatches:
        if capture is not None:
            capture.clear()
        with _autocast(torch, device, autocast_dtype):
            output = model(input_ids=batch, labels=batch)
            task_loss = output.loss
            pressure_loss = activation_l1(capture.activations) if pressure.enabled else None
            loss = task_loss if pressure_loss is None else task_loss + pressure.weight * pressure_loss
        _finite_loss(task_loss, "task loss", torch)
        _finite_loss(loss, "optimization loss", torch)
        if pressure_loss is not None:
            task_grads = accumulate_grads(
                task_grads,
                torch.autograd.grad(task_loss / count, parameters, retain_graph=True, allow_unused=True),
            )
            pressure_grads = accumulate_grads(
                pressure_grads,
                torch.autograd.grad(pressure_loss / count, parameters, retain_graph=True, allow_unused=True),
            )
            pressure_total += float(pressure_loss.detach().cpu())
        (loss / count).backward()
        task_total += float(task_loss.detach().cpu())
        if observer is not None and capture is not None:
            observer(capture.activations)

    result: dict[str, Any] = {"task_loss": task_total / count}
    if pressure.enabled:
        pressure_mean = pressure_total / count
        result.update(gradient_interaction(task_grads, pressure_grads, eps=pressure.eps))
        result.update(
            pressure_loss=pressure_mean,
            pressure_weight=pressure.weight,
            weighted_pressure_loss=pressure.weight * pressure_mean,
            augmented_loss=task_total / count + pressure.weight * pressure_mean,
        )
    result.update(
        prepare_adamw_gradients(
            parameters, gradient_clip_norm=gradient_clip_norm, torch=torch
        )
    )
    optimizer.step()
    return result


def _ol1_boundary(
    *,
    model: Any,
    optimizer: Any,
    parameters: list[Any],
    microbatches: tuple[Any, ...],
    pressure: PressureConfig,
    capture: Any,
    torch: Any,
    device: Any,
    autocast_dtype: Any | None,
    gradient_clip_norm: float | None,
    observer: Callable[[dict[str, Any]], None] | None,
) -> dict[str, Any]:
    optimizer.zero_grad(set_to_none=True)
    task_total = pressure_total = 0.0
    pressure_grads: list[Any | None] = []
    count = len(microbatches)
    for batch in microbatches:
        capture.clear()
        with _autocast(torch, device, autocast_dtype):
            output = model(input_ids=batch, labels=batch)
            task_loss = output.loss
            pressure_loss = activation_l1(capture.activations)
        _finite_loss(task_loss, "task loss", torch)
        _finite_loss(pressure_loss, "pressure loss", torch)
        (task_loss / count).backward(retain_graph=True)
        pressure_grads = accumulate_grads(
            pressure_grads,
            torch.autograd.grad(pressure_loss / count, parameters, allow_unused=True),
        )
        task_total += float(task_loss.detach().cpu())
        pressure_total += float(pressure_loss.detach().cpu())
        if observer is not None:
            observer(capture.activations)

    clip = prepare_adamw_gradients(
        parameters, gradient_clip_norm=gradient_clip_norm, torch=torch
    )
    task_grads = clone_grads(parameters)
    result = {
        "task_loss": task_total / count,
        "pressure_loss": pressure_total / count,
        "pressure_weight": pressure.weight,
        "weighted_pressure_loss": pressure.weight * pressure_total / count,
        "augmented_loss": task_total / count + pressure.weight * pressure_total / count,
        **gradient_interaction(task_grads, pressure_grads, eps=pressure.eps),
        **clip,
    }
    optimizer.step()
    result.update(
        apply_ol1_correction(
            optimizer,
            parameters,
            task_grads,
            pressure_grads,
            pressure_weight=pressure.weight,
            step_budget=float(pressure.step_budget),
            eps=pressure.eps,
        )
    )
    return result


def prepare_adamw_gradients(
    parameters: list[Any], *, gradient_clip_norm: float | None, torch: Any
) -> dict[str, float | bool | None]:
    """Validate accumulated gradients and optionally apply one global L2 clip."""

    if gradient_clip_norm is None:
        norm = parameter_gradient_norm(parameters)
        if not math.isfinite(norm):
            raise RuntimeError("Non-finite accumulated gradient norm.")
        return {
            "adamw_gradient_norm_pre_clip": norm,
            "adamw_gradient_norm_post_clip": norm,
            "adamw_gradient_clip_norm": None,
            "adamw_gradient_clipping_enabled": False,
            "adamw_gradient_was_clipped": False,
        }
    return clip_adamw_gradients(
        parameters, max_norm=float(gradient_clip_norm), torch=torch
    )


def clip_adamw_gradients(
    parameters: list[Any], *, max_norm: float = GLOBAL_GRADIENT_CLIP_NORM, torch: Any
) -> dict[str, float | bool | None]:
    if not math.isfinite(max_norm) or max_norm <= 0.0:
        raise ValueError("Gradient clip norm must be positive and finite, or None.")
    pre = torch.nn.utils.clip_grad_norm_(
        parameters,
        max_norm=max_norm,
        norm_type=2.0,
        error_if_nonfinite=True,
    )
    pre_norm = float(pre.detach().float().cpu())
    post_norm = parameter_gradient_norm(parameters)
    return {
        "adamw_gradient_norm_pre_clip": pre_norm,
        "adamw_gradient_norm_post_clip": post_norm,
        "adamw_gradient_clip_norm": max_norm,
        "adamw_gradient_clipping_enabled": True,
        "adamw_gradient_was_clipped": pre_norm > max_norm,
    }


def parameter_gradient_norm(parameters: list[Any]) -> float:
    squared = 0.0
    for parameter in parameters:
        if parameter.grad is None:
            continue
        norm = float(parameter.grad.detach().float().norm(2).cpu())
        squared += norm * norm
    return squared**0.5


def warmup_steps(max_steps: int, fraction: float = 0.01) -> int:
    if max_steps <= 0 or not 0.0 < fraction <= 1.0:
        raise ValueError("Steps and warmup fraction must be positive.")
    return math.ceil(max_steps * fraction)


def learning_rate(
    step: int,
    *,
    peak: float,
    max_steps: int,
    warmup: int,
    minimum_ratio: float = 0.1,
) -> float:
    if not 1 <= step <= max_steps or not 1 <= warmup <= max_steps:
        raise ValueError("Step and warmup must lie inside the training budget.")
    if step <= warmup:
        return peak * step / warmup
    if step == max_steps:
        return peak * minimum_ratio
    progress = (step - warmup) / (max_steps - warmup)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return peak * (minimum_ratio + (1.0 - minimum_ratio) * cosine)


def set_learning_rate(optimizer: Any, value: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = float(value)


def _autocast(torch: Any, device: Any, dtype: Any | None) -> Any:
    return torch.autocast(device_type=device.type, dtype=dtype) if dtype is not None and device.type == "cuda" else nullcontext()


def _finite_loss(loss: Any, label: str, torch: Any) -> None:
    if not bool(torch.isfinite(loss.detach()).item()):
        raise RuntimeError(f"Non-finite {label}.")

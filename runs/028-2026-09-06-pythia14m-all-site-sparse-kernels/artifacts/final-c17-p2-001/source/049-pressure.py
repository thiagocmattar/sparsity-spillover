"""L1 activation pressure, gradient interaction, and OL1 correction math."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import torch

from .sites import resolve_sites


PRESSURE_METHODS = frozenset({"none", "l1_naive", "orthogonal_l1"})


@dataclass(frozen=True)
class PressureConfig:
    method: str
    sites: tuple[str, ...]
    weight: float
    step_budget: float | None = None
    eps: float = 1e-12

    @property
    def enabled(self) -> bool:
        return self.method != "none"

    @property
    def orthogonal(self) -> bool:
        return self.method == "orthogonal_l1"


def parse_pressure_config(raw: dict[str, Any]) -> PressureConfig:
    method = raw.get("method")
    if method not in PRESSURE_METHODS:
        raise ValueError(f"Unsupported pressure method: {method!r}.")
    sites = resolve_sites(tuple(raw.get("sites", ())), allow_empty=method == "none")
    weight = _finite(raw.get("weight"), "pressure weight")
    if weight < 0.0 or (method == "none" and weight != 0.0):
        raise ValueError("Pressure weight must be nonnegative and zero for method none.")
    step_budget_raw = raw.get("step_budget")
    step_budget = None if step_budget_raw is None else _finite(step_budget_raw, "step budget")
    if method == "orthogonal_l1":
        if step_budget is None or step_budget <= 0.0:
            raise ValueError("OL1 requires a positive step budget.")
    elif step_budget is not None:
        raise ValueError("Only OL1 accepts a step budget.")
    eps = _finite(raw.get("eps", 1e-12), "pressure epsilon")
    if eps <= 0.0:
        raise ValueError("Pressure epsilon must be positive.")
    return PressureConfig(method, sites, weight, step_budget, eps)


def activation_l1(activations: dict[str, Any]) -> Any:
    """Mean within each tensor, then unweighted mean across tensors."""

    if not activations:
        raise ValueError("No captured activations are available for L1 pressure.")
    return torch.stack([value.float().abs().mean() for value in activations.values()]).mean()


def clone_grads(parameters: list[Any]) -> list[Any | None]:
    return [None if parameter.grad is None else parameter.grad.detach().clone() for parameter in parameters]


def accumulate_grads(
    accumulated: list[Any | None],
    new: tuple[Any | None, ...],
) -> list[Any | None]:
    if not accumulated:
        return [None if gradient is None else gradient.detach().clone() for gradient in new]
    if len(accumulated) != len(new):
        raise ValueError("Gradient lists must have equal length.")
    for index, gradient in enumerate(new):
        if gradient is None:
            continue
        if accumulated[index] is None:
            accumulated[index] = gradient.detach().clone()
        else:
            accumulated[index].add_(gradient.detach())
    return accumulated


def gradient_interaction(
    task_grads: list[Any | None], pressure_grads: list[Any | None], *, eps: float = 1e-12
) -> dict[str, float | bool]:
    if len(task_grads) != len(pressure_grads):
        raise ValueError("Task and pressure gradient lists must have equal length.")
    task_sq = pressure_sq = dot = 0.0
    for task, pressure in zip(task_grads, pressure_grads, strict=True):
        if task is not None:
            task_sq += float(task.detach().float().square().sum())
        if pressure is not None:
            pressure_sq += float(pressure.detach().float().square().sum())
        if task is not None and pressure is not None:
            dot += float((task.detach().float() * pressure.detach().float()).sum())
    task_norm = task_sq**0.5
    pressure_norm = pressure_sq**0.5
    return {
        "task_gradient_norm": task_norm,
        "pressure_gradient_norm": pressure_norm,
        "pressure_to_task_gradient_norm_ratio": pressure_norm / (task_norm + eps),
        "task_pressure_gradient_dot": dot,
        "task_pressure_gradient_cosine": dot / (task_norm * pressure_norm + eps),
        "gradient_conflict": dot < 0.0,
    }


@torch.no_grad()
def apply_ol1_correction(
    optimizer: Any,
    parameters: list[Any],
    task_grads: list[Any | None],
    pressure_grads: list[Any | None],
    *,
    pressure_weight: float,
    step_budget: float,
    eps: float = 1e-12,
) -> dict[str, float | bool | int]:
    """Apply the post-AdamW OL1 correction using task-only Adam moments."""

    if not (len(parameters) == len(task_grads) == len(pressure_grads)):
        raise ValueError("Parameters and gradient lists must have equal length.")
    if pressure_weight < 0.0 or step_budget <= 0.0 or eps <= 0.0:
        raise ValueError("OL1 weight must be nonnegative; budget and epsilon must be positive.")

    parameter_index = {id(parameter): index for index, parameter in enumerate(parameters)}
    directions: list[tuple[Any, Any, Any, float]] = []
    task_sq = pressure_sq = dot_before = 0.0
    skipped = 0

    for group in optimizer.param_groups:
        learning_rate = float(group["lr"])
        beta1, beta2 = (float(value) for value in group.get("betas", (0.9, 0.999)))
        adam_eps = float(group.get("eps", 1e-8))
        for parameter in group["params"]:
            index = parameter_index.get(id(parameter))
            if index is None or task_grads[index] is None or pressure_grads[index] is None:
                skipped += 1
                continue
            state = optimizer.state.get(parameter, {})
            if not {"step", "exp_avg", "exp_avg_sq"}.issubset(state):
                skipped += 1
                continue
            step = _as_float(state["step"])
            correction1, correction2 = 1.0 - beta1**step, 1.0 - beta2**step
            if step <= 0.0 or correction1 <= 0.0 or correction2 <= 0.0:
                skipped += 1
                continue
            denominator = state["exp_avg_sq"].detach().float().div(correction2).sqrt().add(adam_eps)
            task_direction = state["exp_avg"].detach().float().div(correction1).div(denominator)
            pressure_direction = pressure_grads[index].detach().float().div(denominator)
            task_sq += float(task_direction.square().sum())
            pressure_sq += float(pressure_direction.square().sum())
            dot_before += float((task_direction * pressure_direction).sum())
            directions.append((parameter, task_direction, pressure_direction, learning_rate))

    projected = dot_before < 0.0 and task_sq > eps
    coefficient = dot_before / (task_sq + eps) if projected else 0.0
    safe: list[tuple[Any, Any, float]] = []
    safe_sq = dot_after = 0.0
    for parameter, task_direction, pressure_direction, learning_rate in directions:
        safe_direction = pressure_direction - coefficient * task_direction if projected else pressure_direction
        safe_sq += float(safe_direction.square().sum())
        dot_after += float((task_direction * safe_direction).sum())
        safe.append((parameter, safe_direction, learning_rate))

    task_norm, pressure_norm, safe_norm = task_sq**0.5, pressure_sq**0.5, safe_sq**0.5
    raw_ratio = pressure_weight * safe_norm / (task_norm + eps)
    scale = min(1.0, step_budget / (raw_ratio + eps)) if raw_ratio > 0.0 else 1.0
    for parameter, direction, learning_rate in safe:
        parameter.add_(
            direction.to(device=parameter.device, dtype=parameter.dtype),
            alpha=-learning_rate * pressure_weight * scale,
        )
    return {
        "task_direction_norm": task_norm,
        "pressure_direction_norm_raw": pressure_norm,
        "task_pressure_dot_before": dot_before,
        "task_pressure_cosine_before": dot_before / (task_norm * pressure_norm + eps),
        "projection_applied": projected,
        "task_pressure_dot_after": dot_after,
        "task_pressure_cosine_after": dot_after / (task_norm * safe_norm + eps),
        "pressure_to_task_ratio_raw": raw_ratio,
        "trust_scale": scale,
        "pressure_to_task_ratio_final": raw_ratio * scale,
        "eligible_parameter_tensors": len(directions),
        "skipped_parameter_tensors": skipped,
    }


def _as_float(value: Any) -> float:
    return float(value.detach().cpu().item()) if hasattr(value, "detach") else float(value)


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{label} must be finite.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite.")
    return result


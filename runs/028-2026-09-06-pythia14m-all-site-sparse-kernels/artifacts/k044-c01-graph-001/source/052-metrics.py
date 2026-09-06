"""Count-first activation, weight, and logical-product diagnostics."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
import math
from typing import Any


LOGICAL_OPERATIONS = (
    "qkv_projection",
    "qk_scores",
    "probability_value",
    "attention_output_projection",
    "mlp_w1",
    "mlp_w2",
)


@dataclass
class _ActivationMoments:
    total: int = 0
    finite: int = 0
    nonfinite: int = 0
    exact_zero: int = 0
    threshold_hits: dict[float, int] = field(default_factory=dict)
    sum: float = 0.0
    square_sum: float = 0.0
    absolute_sum: float = 0.0


class ActivationAccumulator:
    """Stream per-site/layer counts and moments across forward passes."""

    def __init__(self, thresholds: tuple[float, ...] = (0.0, 0.001, 0.01)) -> None:
        values = tuple(float(value) for value in thresholds)
        if any(not math.isfinite(value) or value < 0.0 for value in values):
            raise ValueError("Near-zero thresholds must be finite and nonnegative.")
        if tuple(sorted(set(values))) != values:
            raise ValueError("Near-zero thresholds must be strictly increasing.")
        self.thresholds = values
        self._rows: dict[str, _ActivationMoments] = {}

    def update(self, activations: Mapping[str, Any], *, torch: Any) -> None:
        for name, value in activations.items():
            flat = value.detach().float().reshape(-1)
            finite_mask = torch.isfinite(flat)
            finite_values = flat[finite_mask]
            row = self._rows.setdefault(
                name,
                _ActivationMoments(
                    threshold_hits={threshold: 0 for threshold in self.thresholds}
                ),
            )
            row.total += int(flat.numel())
            row.nonfinite += int((~finite_mask).sum().cpu())
            if not finite_values.numel():
                continue
            absolute = finite_values.abs()
            row.finite += int(finite_values.numel())
            row.exact_zero += int((finite_values == 0).sum().cpu())
            row.sum += float(finite_values.sum().cpu())
            row.square_sum += float(finite_values.square().sum().cpu())
            row.absolute_sum += float(absolute.sum().cpu())
            for threshold in self.thresholds:
                row.threshold_hits[threshold] += int((absolute <= threshold).sum().cpu())

    def rows(self) -> list[dict[str, Any]]:
        return [_activation_row(name, self._rows[name]) for name in sorted(self._rows, key=_site_layer_key)]

    def pooled_by_site(self) -> list[dict[str, Any]]:
        pooled: dict[str, _ActivationMoments] = {}
        for name, row in self._rows.items():
            alias = name.split(".layer_", 1)[0]
            target = pooled.setdefault(
                alias,
                _ActivationMoments(threshold_hits={threshold: 0 for threshold in self.thresholds}),
            )
            _merge_moments(target, row)
        return [_activation_row(alias, pooled[alias]) for alias in sorted(pooled)]


def weight_statistics(
    model: Any,
    *,
    include: Callable[[str, Any], bool] | None = None,
) -> list[dict[str, Any]]:
    """Return named FP32 L2 statistics; inclusion rules remain caller-visible."""

    rows = []
    for name, parameter in model.named_parameters():
        if include is not None and not include(name, parameter):
            continue
        detached = parameter.detach().float().reshape(-1)
        finite = detached.isfinite()
        finite_values = detached[finite]
        square_sum = float(finite_values.square().sum().cpu())
        rows.append(
            {
                "name": name,
                "layer": _layer_from_parameter_name(name),
                "role": _parameter_role(name),
                "elements": int(detached.numel()),
                "finite": int(finite.sum().cpu()),
                "nonfinite": int((~finite).sum().cpu()),
                "square_sum": square_sum,
                "l2_norm": square_sum**0.5,
                "rms": (square_sum / int(finite.sum().cpu())) ** 0.5 if finite_values.numel() else None,
            }
        )
    return rows


def pool_weight_norm(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    square_sum = sum(float(row["square_sum"]) for row in rows)
    elements = sum(int(row["elements"]) for row in rows)
    finite = sum(int(row["finite"]) for row in rows)
    return {
        "parameter_tensors": len(rows),
        "elements": elements,
        "finite": finite,
        "square_sum": square_sum,
        "l2_norm": square_sum**0.5,
        "rms": (square_sum / finite) ** 0.5 if finite else None,
    }


def linear_zero_product_counts(value: Any, *, output_features: int, torch: Any) -> tuple[int, int]:
    detached = value.detach()
    zero_inputs = int(torch.count_nonzero(detached == 0).cpu())
    return zero_inputs * int(output_features), int(detached.numel()) * int(output_features)


def qk_zero_product_counts(query: Any, key: Any, *, torch: Any) -> tuple[int, int]:
    if query.shape != key.shape or query.ndim != 4:
        raise ValueError("QK counting expects matching [B,H,T,d] tensors.")
    batch, heads, tokens, width = query.shape
    query_nonzero = (query.detach() != 0).to(torch.int64)
    cumulative_key_nonzero = (key.detach() != 0).to(torch.int64).cumsum(dim=-2)
    nonzero_products = int((query_nonzero * cumulative_key_nonzero).sum().cpu())
    total = int(batch * heads * width * tokens * (tokens + 1) // 2)
    return total - nonzero_products, total


def pv_zero_product_counts(
    probabilities: Any,
    value: Any,
    *,
    torch: Any,
    query_chunk_size: int = 128,
) -> tuple[int, int]:
    if probabilities.ndim != 4 or value.ndim != 4:
        raise ValueError("PV counting expects rank-four tensors.")
    batch, heads, queries, keys = probabilities.shape
    value_batch, value_heads, value_keys, width = value.shape
    if (batch, heads, keys) != (value_batch, value_heads, value_keys) or queries != keys:
        raise ValueError("PV counting expects matching uncached causal-attention shapes.")
    key_positions = torch.arange(keys, device=probabilities.device)
    value_nonzero_dimensions = torch.count_nonzero(value.detach(), dim=-1).to(torch.int64)
    nonzero_products = 0
    for start in range(0, queries, query_chunk_size):
        stop = min(start + query_chunk_size, queries)
        valid = key_positions.unsqueeze(0) <= torch.arange(start, stop, device=probabilities.device).unsqueeze(1)
        probability_nonzero = probabilities[..., start:stop, :].detach() != 0
        nonzero_products += int(
            ((probability_nonzero & valid).to(torch.int64) * value_nonzero_dimensions.unsqueeze(-2)).sum().cpu()
        )
    total = int(batch * heads * width * queries * (queries + 1) // 2)
    return total - nonzero_products, total


def summarize_logical_products(
    zero_counts: Mapping[str, int],
    product_counts: Mapping[str, int],
    *,
    lm_head_product_count: int,
) -> dict[str, Any]:
    if set(zero_counts) != set(LOGICAL_OPERATIONS) or set(product_counts) != set(LOGICAL_OPERATIONS):
        raise ValueError("Logical counters must contain exactly the six declared block operations.")
    per_operation = {}
    for name in LOGICAL_OPERATIONS:
        zero, total = int(zero_counts[name]), int(product_counts[name])
        if total <= 0 or not 0 <= zero <= total:
            raise ValueError(f"Invalid logical counter for {name}.")
        per_operation[name] = {
            "zero_product_count": zero,
            "product_count": total,
            "zero_product_fraction": zero / total,
        }
    block_zero = sum(row["zero_product_count"] for row in per_operation.values())
    block_total = sum(row["product_count"] for row in per_operation.values())
    lm_head = int(lm_head_product_count)
    if lm_head < 0:
        raise ValueError("LM-head product count must be nonnegative.")
    return {
        "R_block": block_zero / block_total,
        "R_model": block_zero / (block_total + lm_head),
        "block_zero_product_count": block_zero,
        "block_product_count": block_total,
        "lm_head_product_count": lm_head,
        "model_product_count": block_total + lm_head,
        "per_operation": per_operation,
    }


def _activation_row(name: str, row: _ActivationMoments) -> dict[str, Any]:
    return {
        "name": name,
        "total": row.total,
        "finite": row.finite,
        "nonfinite": row.nonfinite,
        "exact_zero_count": row.exact_zero,
        "exact_zero_fraction": row.exact_zero / row.total if row.total else None,
        "threshold_hits": {f"{threshold:g}": row.threshold_hits[threshold] for threshold in sorted(row.threshold_hits)},
        "threshold_fractions": {
            f"{threshold:g}": row.threshold_hits[threshold] / row.total if row.total else None
            for threshold in sorted(row.threshold_hits)
        },
        "sum": row.sum,
        "square_sum": row.square_sum,
        "absolute_sum": row.absolute_sum,
        "mean": row.sum / row.finite if row.finite else None,
        "mean_abs": row.absolute_sum / row.finite if row.finite else None,
        "rms": (row.square_sum / row.finite) ** 0.5 if row.finite else None,
        "l2_norm": row.square_sum**0.5,
    }


def _merge_moments(target: _ActivationMoments, source: _ActivationMoments) -> None:
    target.total += source.total
    target.finite += source.finite
    target.nonfinite += source.nonfinite
    target.exact_zero += source.exact_zero
    target.sum += source.sum
    target.square_sum += source.square_sum
    target.absolute_sum += source.absolute_sum
    for threshold, value in source.threshold_hits.items():
        target.threshold_hits[threshold] += value


def _site_layer_key(name: str) -> tuple[str, int]:
    alias, _, suffix = name.partition(".layer_")
    return alias, int(suffix) if suffix.isdigit() else 10**9


def _layer_from_parameter_name(name: str) -> int | None:
    parts = name.split(".")
    try:
        index = parts.index("layers")
        return int(parts[index + 1])
    except (ValueError, IndexError):
        return None


def _parameter_role(name: str) -> str:
    for suffix, role in (
        (".attention.query_key_value.weight", "qkv_projection"),
        (".attention.dense.weight", "attention_output_projection"),
        (".mlp.dense_h_to_4h.weight", "mlp_w1"),
        (".mlp.dense_4h_to_h.weight", "mlp_w2"),
        (".embed_in.weight", "input_embedding"),
        (".embed_out.weight", "lm_head"),
    ):
        if name.endswith(suffix):
            return role
    if name.endswith(".bias"):
        return "bias"
    if "layernorm" in name.lower() or "layer_norm" in name.lower():
        return "normalization"
    return "other"

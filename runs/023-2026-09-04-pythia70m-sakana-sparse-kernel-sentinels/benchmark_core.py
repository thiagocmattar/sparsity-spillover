"""Count-first occupancy, exact ELL, coverage, and timing helpers for Run 023."""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import Any


LINEAR_OPERATION_SPECS = {
    "qkv_projection": {"site": "a", "K": 512, "N": 1536},
    "mlp_w1": {"site": "m", "K": 512, "N": 2048},
    "mlp_w2": {"site": "h", "K": 2048, "N": 512},
    "attention_output_projection": {"site": "z", "K": 512, "N": 512},
}


def pack_exact_ell_reference(matrix: Any, *, torch: Any) -> tuple[Any, Any, Any]:
    """Reference pack every signed nonzero into a full-capacity ELL buffer."""

    if matrix.ndim != 2 or matrix.shape[0] <= 0 or not 0 < matrix.shape[1] <= 65535:
        raise ValueError("ELL input must be nonempty rank 2 with K <= 65535.")
    matrix = matrix.contiguous()
    rows, columns = matrix.shape
    mask = matrix != 0
    counts = mask.sum(dim=1, dtype=torch.int32)
    values = torch.empty_like(matrix)
    integer_indices = torch.empty((rows, columns), dtype=torch.int32, device=matrix.device)
    nz_rows, nz_columns = mask.nonzero(as_tuple=True)
    if nz_rows.numel():
        positions = mask.to(torch.int32).cumsum(dim=1)[nz_rows, nz_columns] - 1
        values[nz_rows, positions] = matrix[nz_rows, nz_columns]
        integer_indices[nz_rows, positions] = nz_columns.to(torch.int32)
    return values.contiguous(), integer_indices.to(torch.uint16).contiguous(), counts.contiguous()


def unpack_exact_ell(
    values: Any,
    indices: Any,
    counts: Any,
    *,
    columns: int,
    torch: Any,
) -> Any:
    if values.ndim != 2 or indices.shape != values.shape or counts.shape != (values.shape[0],):
        raise ValueError("Malformed ELL tensors.")
    if values.shape[1] < columns:
        raise ValueError("Full-capacity exact ELL requires stride >= source width.")
    dense = torch.zeros((values.shape[0], columns), dtype=values.dtype, device=values.device)
    valid = torch.arange(values.shape[1], device=values.device).unsqueeze(0) < counts.unsqueeze(1)
    rows = torch.arange(values.shape[0], device=values.device).unsqueeze(1).expand_as(valid)
    # CUDA PyTorch does not implement boolean indexing directly for UInt16.
    # Cast before applying the mask; this changes only the Python verifier.
    dense[rows[valid], indices.to(torch.long)[valid]] = values[valid]
    return dense


def validate_exact_ell(
    matrix: Any,
    values: Any,
    indices: Any,
    counts: Any,
    *,
    torch: Any,
) -> dict[str, int | bool]:
    rows, columns = matrix.shape
    if values.shape != (rows, columns) or indices.shape != (rows, columns):
        raise ValueError("Exact ELL buffers must use full K capacity.")
    if counts.shape != (rows,):
        raise ValueError("Exact ELL counts must have one entry per row.")
    if values.dtype != torch.bfloat16 or indices.dtype != torch.uint16 or counts.dtype != torch.int32:
        raise TypeError("Expected BF16 values, uint16 columns, and int32 counts.")
    if not all(tensor.is_contiguous() for tensor in (matrix, values, indices, counts)):
        raise ValueError("Exact ELL tensors must be contiguous.")
    minimum = int(counts.min().item())
    maximum = int(counts.max().item())
    if minimum < 0 or maximum > columns:
        raise ValueError("ELL row count exceeds its full-capacity buffer.")
    reconstructed = unpack_exact_ell(values, indices, counts, columns=columns, torch=torch)
    exact = bool(torch.equal(matrix, reconstructed))
    if not exact:
        raise ValueError("Exact ELL signed round trip failed.")
    expected_nonzeros = int((matrix != 0).sum().item())
    actual_nonzeros = int(counts.sum().item())
    if expected_nonzeros != actual_nonzeros:
        raise ValueError("ELL pack dropped or duplicated nonzeros.")
    return {
        "rows": rows,
        "columns": columns,
        "stride": columns,
        "minimum_row_nnz": minimum,
        "maximum_row_nnz": maximum,
        "nonzeros": actual_nonzeros,
        "signed_round_trip_exact": exact,
        "dropped_values": 0,
    }


def _add_histogram(destination: list[int], source: Any) -> None:
    values = source.detach().cpu().tolist()
    if len(destination) < len(values):
        destination.extend([0] * (len(values) - len(destination)))
    for index, value in enumerate(values):
        destination[index] += int(value)


def histogram_quantile(histogram: list[int], quantile: float) -> int:
    total = sum(histogram)
    if not 0 <= quantile <= 1 or total <= 0:
        raise ValueError("Histogram quantile requires q in [0,1] and nonempty data.")
    rank = max(1, math.ceil(quantile * total))
    cumulative = 0
    for value, count in enumerate(histogram):
        cumulative += count
        if cumulative >= rank:
            return value
    raise AssertionError("Unreachable histogram quantile.")


def _histogram_summary(histogram: list[int]) -> dict[str, Any]:
    total = sum(histogram)
    return {
        "count": total,
        "mean": sum(value * count for value, count in enumerate(histogram)) / total,
        "min": histogram_quantile(histogram, 0.0),
        "p50": histogram_quantile(histogram, 0.5),
        "p90": histogram_quantile(histogram, 0.9),
        "p99": histogram_quantile(histogram, 0.99),
        "max": histogram_quantile(histogram, 1.0),
        "histogram": histogram,
    }


@dataclass
class OccupancyAccumulator:
    width: int
    tile_width: int
    near_zero_thresholds: tuple[float, ...]
    payload_capacities: dict[str, int]
    elements: int = 0
    rows: int = 0
    exact_zeros: int = 0
    near_zeros: dict[float, int] = field(default_factory=dict)
    sum_values: float = 0.0
    sum_squares: float = 0.0
    row_l2_sum: float = 0.0
    row_nnz_histogram: list[int] = field(default_factory=list)
    tile_nnz_histogram: list[int] = field(default_factory=list)
    overflow_tiles: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.width <= 0 or self.tile_width <= 0 or self.width % self.tile_width:
            raise ValueError("Width must be a positive multiple of tile width.")
        if any(threshold <= 0 for threshold in self.near_zero_thresholds):
            raise ValueError("Near-zero thresholds must be positive.")
        self.near_zeros = {threshold: 0 for threshold in self.near_zero_thresholds}
        self.overflow_tiles = {name: 0 for name in self.payload_capacities}

    def update(self, activation: Any, *, torch: Any) -> None:
        if activation.shape[-1] != self.width:
            raise ValueError(f"Expected final width {self.width}, got {activation.shape[-1]}.")
        flat = activation.detach().reshape(-1, self.width).float()
        if not bool(torch.isfinite(flat).all().item()):
            raise RuntimeError("Non-finite captured activation.")
        absolute = flat.abs()
        nonzero = flat != 0
        row_nnz = nonzero.sum(dim=1)
        tile_nnz = nonzero.reshape(-1, self.width // self.tile_width, self.tile_width).sum(dim=2)
        self.elements += flat.numel()
        self.rows += flat.shape[0]
        self.exact_zeros += int((~nonzero).sum().item())
        for threshold in self.near_zero_thresholds:
            self.near_zeros[threshold] += int((absolute <= threshold).sum().item())
        self.sum_values += float(flat.sum(dtype=torch.float64).item())
        self.sum_squares += float(flat.square().sum(dtype=torch.float64).item())
        self.row_l2_sum += float(flat.square().sum(dim=1).sqrt().sum(dtype=torch.float64).item())
        _add_histogram(self.row_nnz_histogram, torch.bincount(row_nnz.cpu(), minlength=self.width + 1))
        _add_histogram(self.tile_nnz_histogram, torch.bincount(tile_nnz.reshape(-1).cpu(), minlength=self.tile_width + 1))
        for name, capacity in self.payload_capacities.items():
            self.overflow_tiles[name] += int((tile_nnz > capacity).sum().item())

    def finalize(self) -> dict[str, Any]:
        if not self.elements:
            raise ValueError("Cannot finalize an empty accumulator.")
        tiles = self.rows * (self.width // self.tile_width)
        return {
            "elements": self.elements,
            "rows": self.rows,
            "tiles": tiles,
            "exact_zero_count": self.exact_zeros,
            "exact_zero_mass": self.exact_zeros / self.elements,
            "near_zero": {
                str(threshold): {"count": count, "mass": count / self.elements}
                for threshold, count in self.near_zeros.items()
            },
            "mean": self.sum_values / self.elements,
            "rms": math.sqrt(self.sum_squares / self.elements),
            "global_l2": math.sqrt(self.sum_squares),
            "mean_row_l2": self.row_l2_sum / self.rows,
            "row_nnz": _histogram_summary(self.row_nnz_histogram),
            "tile_nnz": _histogram_summary(self.tile_nnz_histogram),
            "tile_overflow": {
                name: {"count": count, "mass": count / tiles, "payload_capacity": capacity}
                for name, capacity in self.payload_capacities.items()
                for count in (self.overflow_tiles[name],)
            },
            "integer_pooling": True,
        }


@dataclass
class AttentionOpportunityAccumulator:
    """Lower-bound attention opportunity covered by q-only and v-only packing."""

    sequence_length: int
    q_only_causal_zero_products: int = 0
    v_only_causal_zero_products: int = 0
    q_only_physical_full_gemm_zero_products: int = 0
    v_only_physical_full_gemm_zero_products: int = 0
    causal_products: int = 0
    physical_full_gemm_products: int = 0
    sequences: int = 0

    def update(self, query: Any, value: Any, *, torch: Any) -> None:
        if query.ndim != 4 or value.shape != query.shape:
            raise ValueError("Expected q and v in [B,H,T,d] with equal shapes.")
        batch, heads, sequence, head_size = query.shape
        if sequence != self.sequence_length:
            raise ValueError("Attention opportunity requires complete fixed-length blocks.")
        q_zeros = query.detach() == 0
        v_zeros = value.detach() == 0
        q_by_position = q_zeros.sum(dim=(0, 1, 3), dtype=torch.int64)
        v_by_position = v_zeros.sum(dim=(0, 1, 3), dtype=torch.int64)
        q_weights = torch.arange(1, sequence + 1, dtype=torch.int64, device=query.device)
        v_weights = torch.arange(sequence, 0, -1, dtype=torch.int64, device=query.device)
        self.q_only_causal_zero_products += int((q_by_position * q_weights).sum().item())
        self.v_only_causal_zero_products += int((v_by_position * v_weights).sum().item())
        self.q_only_physical_full_gemm_zero_products += int(q_zeros.sum().item()) * sequence
        self.v_only_physical_full_gemm_zero_products += int(v_zeros.sum().item()) * sequence
        self.causal_products += batch * heads * head_size * sequence * (sequence + 1) // 2
        self.physical_full_gemm_products += batch * heads * head_size * sequence * sequence
        self.sequences += batch

    def finalize(self) -> dict[str, Any]:
        if not self.sequences:
            raise ValueError("Cannot finalize empty attention opportunity.")
        return {
            "sequences": self.sequences,
            "q_only_causal_zero_products": self.q_only_causal_zero_products,
            "v_only_causal_zero_products": self.v_only_causal_zero_products,
            "q_only_physical_full_gemm_zero_products": self.q_only_physical_full_gemm_zero_products,
            "v_only_physical_full_gemm_zero_products": self.v_only_physical_full_gemm_zero_products,
            "causal_products_per_operation": self.causal_products,
            "physical_full_gemm_products_per_operation": self.physical_full_gemm_products,
            "q_only_causal_fraction": self.q_only_causal_zero_products / self.causal_products,
            "v_only_causal_fraction": self.v_only_causal_zero_products / self.causal_products,
            "lower_bound_not_qk_union": True,
            "integer_pooling": True,
        }


def covered_opportunity(
    logical_products: dict[str, Any],
    linear_operations: list[str],
    *,
    attention: dict[str, Any] | None = None,
) -> dict[str, Any]:
    measured = logical_products["measured"]
    canonical_operations = measured["per_operation"]
    linear_counts = {
        operation: int(canonical_operations[operation]["zero_product_count"])
        for operation in linear_operations
    }
    q_only = v_only = 0
    if attention is not None:
        q_only = int(attention["q_only_causal_zero_products"])
        v_only = int(attention["v_only_causal_zero_products"])
        if q_only > int(canonical_operations["qk_scores"]["zero_product_count"]):
            raise ValueError("q-only opportunity cannot exceed canonical Q/K-union opportunity.")
        if v_only > int(canonical_operations["probability_value"]["zero_product_count"]):
            raise ValueError("v-only opportunity cannot exceed canonical P/V-union opportunity.")
    numerator = sum(linear_counts.values()) + q_only + v_only
    denominator = int(measured["model_product_count"])
    canonical_numerator = int(measured["block_zero_product_count"])
    if numerator > canonical_numerator:
        raise ValueError("Kernel-covered opportunity cannot exceed canonical zero products.")
    return {
        "covered_zero_product_count": numerator,
        "canonical_model_product_count": denominator,
        "R_covered": numerator / denominator,
        "canonical_R_model": float(measured["R_model"]),
        "linear_zero_products": linear_counts,
        "q_only_causal_zero_products": q_only,
        "v_only_causal_zero_products": v_only,
        "attention_is_lower_bound": attention is not None,
        "not_runtime_speedup": True,
    }


def relative_errors(actual: Any, reference: Any, *, torch: Any) -> dict[str, float]:
    if not bool(torch.isfinite(actual).all().item()) or not bool(torch.isfinite(reference).all().item()):
        raise RuntimeError("Cannot compare non-finite tensors.")
    delta = actual.float() - reference.float()
    denominator = max(float(reference.float().norm().item()), 1e-30)
    return {
        "relative_l2": float(delta.norm().item()) / denominator,
        "maximum_absolute": float(delta.abs().max().item()),
        "mean_absolute": float(delta.abs().mean().item()),
    }


def _quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def summarize_latency(block_ms: list[float], *, batch_size: int, sequence_length: int) -> dict[str, Any]:
    if not block_ms or any(not math.isfinite(value) or value <= 0 for value in block_ms):
        raise ValueError("Latency blocks must be finite and positive.")
    median_ms = statistics.median(block_ms)
    return {
        "block_ms": block_ms,
        "median_ms": median_ms,
        "p10_ms": _quantile(block_ms, 0.1),
        "p90_ms": _quantile(block_ms, 0.9),
        "minimum_ms": min(block_ms),
        "maximum_ms": max(block_ms),
        "sequences_per_second": batch_size * 1000.0 / median_ms,
        "tokens_per_second": batch_size * sequence_length * 1000.0 / median_ms,
    }


def paired_speedup(dense_ms: list[float], candidate_ms: list[float]) -> dict[str, Any]:
    if len(dense_ms) != len(candidate_ms) or not dense_ms:
        raise ValueError("Paired timings require equal nonempty block lists.")
    ratios = [dense / candidate for dense, candidate in zip(dense_ms, candidate_ms, strict=True)]
    return {
        "paired_block_speedups": ratios,
        "median": statistics.median(ratios),
        "p10": _quantile(ratios, 0.1),
        "p90": _quantile(ratios, 0.9),
        "blocks": len(ratios),
    }


def pv_v_left_equivalent(probabilities: Any, value: Any) -> Any:
    """Algebraic reference for the v-packed PV transformation."""

    return (value.transpose(-2, -1) @ probabilities.transpose(-2, -1)).transpose(-2, -1)

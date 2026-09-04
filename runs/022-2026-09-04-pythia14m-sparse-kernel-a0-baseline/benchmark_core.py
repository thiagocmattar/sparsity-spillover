"""Exact ELL packing and count-first activation summaries for Run 022."""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import Any


def _round_up(value: int, alignment: int) -> int:
    if alignment <= 0:
        raise ValueError("alignment must be positive")
    return max(alignment, ((value + alignment - 1) // alignment) * alignment)


def pack_exact_ell(matrix: Any, *, torch: Any, alignment: int = 8) -> tuple[Any, Any, Any, int]:
    """Pack every nonzero of a rank-2 matrix; no pruning or truncation."""

    if matrix.ndim != 2:
        raise ValueError("ELL input must be rank 2.")
    rows, columns = matrix.shape
    if columns > 65535:
        raise ValueError("uint16 ELL column indices require K <= 65535.")
    if not matrix.is_contiguous():
        matrix = matrix.contiguous()
    mask = matrix != 0
    row_counts = mask.sum(dim=1, dtype=torch.int32)
    maximum = int(row_counts.max().item()) if rows else 0
    stride = _round_up(maximum, alignment)
    values = torch.zeros((rows, stride), dtype=matrix.dtype, device=matrix.device)
    integer_indices = torch.zeros((rows, stride), dtype=torch.int32, device=matrix.device)
    nz_rows, nz_columns = mask.nonzero(as_tuple=True)
    if nz_rows.numel():
        positions = mask.to(torch.int32).cumsum(dim=1)[nz_rows, nz_columns] - 1
        values[nz_rows, positions] = matrix[nz_rows, nz_columns]
        integer_indices[nz_rows, positions] = nz_columns.to(torch.int32)
    indices = integer_indices.to(torch.uint16)
    return values.contiguous(), indices.contiguous(), row_counts.contiguous(), stride


def unpack_exact_ell(
    values: Any,
    indices: Any,
    row_counts: Any,
    *,
    columns: int,
    torch: Any,
) -> Any:
    if values.ndim != 2 or indices.shape != values.shape or row_counts.shape != (values.shape[0],):
        raise ValueError("Malformed ELL tensors.")
    dense = torch.zeros((values.shape[0], columns), dtype=values.dtype, device=values.device)
    valid = torch.arange(values.shape[1], device=values.device).unsqueeze(0) < row_counts.unsqueeze(1)
    rows = torch.arange(values.shape[0], device=values.device).unsqueeze(1).expand_as(valid)
    dense[rows[valid], indices[valid].to(torch.long)] = values[valid]
    return dense


def validate_raw_ell_contract(
    values: Any,
    indices: Any,
    row_counts: Any,
    dense_rhs: Any,
    *,
    overflow_threshold: int,
    require_cuda: bool = True,
    torch: Any,
) -> dict[str, int]:
    if values.ndim != 2 or indices.shape != values.shape:
        raise ValueError("ELL values and indices must be equal rank-2 shapes.")
    rows, stride = values.shape
    if rows <= 0 or stride <= 0:
        raise ValueError("The raw ELL kernel requires positive M and stride.")
    if row_counts.shape != (rows,):
        raise ValueError("ELL row_counts must have one entry per row.")
    if dense_rhs.ndim != 2:
        raise ValueError("Dense RHS must be rank 2 [K, N].")
    if dense_rhs.shape[0] <= 0 or dense_rhs.shape[0] > 65535:
        raise ValueError("The raw ELL uint16 format requires K in [1, 65535].")
    if dense_rhs.shape[1] <= 0 or dense_rhs.shape[1] % 8:
        raise ValueError("The raw ELL kernel requires positive N divisible by 8.")
    if values.dtype != torch.bfloat16 or dense_rhs.dtype != torch.bfloat16:
        raise TypeError("The upstream raw ELL kernel requires BF16 values and RHS.")
    if indices.dtype != torch.uint16 or row_counts.dtype != torch.int32:
        raise TypeError("The upstream raw ELL kernel requires uint16 indices and int32 counts.")
    if not all(tensor.is_contiguous() for tensor in (values, indices, row_counts, dense_rhs)):
        raise ValueError("The raw ELL inputs must be contiguous.")
    if len({tensor.device for tensor in (values, indices, row_counts, dense_rhs)}) != 1:
        raise ValueError("All raw ELL inputs must be on the same device.")
    if require_cuda and not all(tensor.is_cuda for tensor in (values, indices, row_counts, dense_rhs)):
        raise ValueError("The raw ELL inputs must be CUDA tensors.")
    maximum = int(row_counts.max().item()) if rows else 0
    if overflow_threshold < maximum:
        raise ValueError("overflow_threshold would silently skip at least one row.")
    if overflow_threshold > stride:
        raise ValueError("overflow_threshold cannot exceed the allocated ELL stride.")
    if int(row_counts.min().item()) < 0 or maximum > stride:
        raise ValueError("ELL row count lies outside the allocated stride.")
    valid = torch.arange(stride, device=values.device).unsqueeze(0) < row_counts.unsqueeze(1)
    if bool(valid.any().item()) and int(indices[valid].to(torch.int32).max().item()) >= dense_rhs.shape[0]:
        raise ValueError("ELL column index lies outside the dense RHS K dimension.")
    return {"M": rows, "K": dense_rhs.shape[0], "N": dense_rhs.shape[1], "stride": stride, "max_row_nnz": maximum}


def _add_histogram(destination: list[int], source: Any) -> None:
    values = source.detach().to(device="cpu", dtype=source.dtype).tolist()
    if len(destination) < len(values):
        destination.extend([0] * (len(values) - len(destination)))
    for index, value in enumerate(values):
        destination[index] += int(value)


def histogram_quantile(histogram: list[int], quantile: float) -> int:
    if not 0.0 <= quantile <= 1.0 or sum(histogram) <= 0:
        raise ValueError("Quantile requires q in [0,1] and a nonempty histogram.")
    rank = max(1, math.ceil(quantile * sum(histogram)))
    cumulative = 0
    for value, count in enumerate(histogram):
        cumulative += count
        if cumulative >= rank:
            return value
    raise AssertionError("Unreachable histogram quantile.")


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
            raise ValueError("Near-zero thresholds must be positive; exact zero is separate.")
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
        if not self.elements or not self.rows:
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
                name: {"count": count, "mass": count / tiles, "payload_capacity": self.payload_capacities[name]}
                for name, count in self.overflow_tiles.items()
            },
            "integer_pooling": True,
        }


def _histogram_summary(histogram: list[int]) -> dict[str, Any]:
    total = sum(histogram)
    weighted = sum(value * count for value, count in enumerate(histogram))
    return {
        "count": total,
        "mean": weighted / total,
        "min": histogram_quantile(histogram, 0.0),
        "p50": histogram_quantile(histogram, 0.5),
        "p90": histogram_quantile(histogram, 0.9),
        "p99": histogram_quantile(histogram, 0.99),
        "max": histogram_quantile(histogram, 1.0),
        "histogram": histogram,
    }


def summarize_latency(block_ms: list[float], *, batch_size: int, sequence_length: int) -> dict[str, Any]:
    if not block_ms or any(not math.isfinite(value) or value <= 0 for value in block_ms):
        raise ValueError("Latency blocks must be finite and positive.")
    ordered = sorted(block_ms)
    median_ms = statistics.median(ordered)
    return {
        "block_ms": block_ms,
        "median_ms": median_ms,
        "minimum_ms": min(ordered),
        "maximum_ms": max(ordered),
        "sequences_per_second": batch_size * 1000.0 / median_ms,
        "tokens_per_second": batch_size * sequence_length * 1000.0 / median_ms,
    }


def relative_errors(actual: Any, reference: Any, *, torch: Any) -> dict[str, float]:
    if not bool(torch.isfinite(actual).all().item()) or not bool(torch.isfinite(reference).all().item()):
        raise RuntimeError("Cannot measure numerical error for non-finite tensors.")
    delta = actual.float() - reference.float()
    denominator = max(float(reference.float().norm().item()), 1e-30)
    return {
        "relative_l2": float(delta.norm().item()) / denominator,
        "maximum_absolute": float(delta.abs().max().item()),
        "mean_absolute": float(delta.abs().mean().item()),
    }

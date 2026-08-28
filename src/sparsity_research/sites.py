"""Canonical activation sites, topology registry, and exact-zero gates."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
from typing import Any

import torch


SITE_ORDER = (
    "a",
    "m",
    "h",
    "q_pre",
    "k_pre",
    "q_post",
    "k_post",
    "v",
)
SUPPORTED_SITES = frozenset(SITE_ORDER)


@dataclass(frozen=True)
class SiteSpec:
    alias: str
    tensor: str
    shape: str
    downstream_matmul: str
    operations_before_matmul: tuple[str, ...] = ()


SITE_SPECS = {
    "a": SiteSpec("a", "attention branch LayerNorm/gate output", "[B,T,D]", "fused QKV projection"),
    "m": SiteSpec("m", "MLP branch LayerNorm/gate output", "[B,T,D]", "MLP W1/up projection"),
    "h": SiteSpec("h", "MLP hidden nonlinearity/gate output", "[B,T,M]", "MLP W2/down projection"),
    "q_pre": SiteSpec("q_pre", "query before partial RoPE", "[B,H,T,d]", "QK scores", ("partial RoPE",)),
    "k_pre": SiteSpec("k_pre", "key before partial RoPE", "[B,H,T,d]", "QK scores", ("partial RoPE",)),
    "q_post": SiteSpec("q_post", "query after partial RoPE", "[B,H,T,d]", "QK scores"),
    "k_post": SiteSpec("k_post", "key after partial RoPE", "[B,H,T,d]", "QK scores"),
    "v": SiteSpec("v", "value from fused QKV projection", "[B,H,T,d]", "probability-value product"),
}


@dataclass(frozen=True)
class Topology:
    topology_id: str
    active_sites: tuple[str, ...]

    @property
    def qk_placement(self) -> str | None:
        if {"q_pre", "k_pre"}.intersection(self.active_sites):
            return "pre_rope"
        if {"q_post", "k_post"}.intersection(self.active_sites):
            return "post_rope"
        return None


_TOPOLOGY_ROWS = (
    ("A0", ()),
    ("A1-H", ("h",)),
    ("A2", ("m", "h")),
    ("A3", ("a", "m", "h")),
    ("A4-Q", ("a", "m", "h", "q_post")),
    ("A4-K", ("a", "m", "h", "k_post")),
    ("A4-V", ("a", "m", "h", "v")),
    ("A5-QK-PRE", ("a", "m", "h", "q_pre", "k_pre")),
    ("A5-QK-POST", ("a", "m", "h", "q_post", "k_post")),
    ("A6-PRE", ("a", "m", "h", "q_pre", "k_pre", "v")),
    ("A6-POST", ("a", "m", "h", "q_post", "k_post", "v")),
)
TOPOLOGIES = {name: Topology(name, sites) for name, sites in _TOPOLOGY_ROWS}


class FixedOneSidedThreshold(torch.nn.Module):
    """Keep values at or above a fixed threshold; equality survives."""

    def __init__(self, kappa: float) -> None:
        super().__init__()
        self.kappa = _nonnegative_finite(kappa, "kappa")

    def forward(self, value: Any) -> Any:
        return value.masked_fill(value.detach() < self.kappa, 0.0)


class FixedSymmetricThreshold(torch.nn.Module):
    """Keep signed values at or beyond a fixed magnitude; equality survives."""

    def __init__(self, kappa: float) -> None:
        super().__init__()
        self.kappa = _nonnegative_finite(kappa, "kappa")

    def forward(self, value: Any) -> Any:
        return value.masked_fill(value.detach().abs() < self.kappa, 0.0)


def resolve_sites(sites: list[str] | tuple[str, ...], *, allow_empty: bool = False) -> tuple[str, ...]:
    values = tuple(sites)
    if not values and not allow_empty:
        raise ValueError("At least one canonical activation site is required.")
    if len(values) != len(set(values)):
        raise ValueError("Activation sites must not contain duplicates.")
    unknown = [site for site in values if site not in SUPPORTED_SITES]
    if unknown:
        raise ValueError("Unknown activation sites: " + ", ".join(unknown))
    return values


def resolve_topology(topology_id: Any) -> Topology:
    if not isinstance(topology_id, str) or topology_id not in TOPOLOGIES:
        raise ValueError(f"Unsupported topology: {topology_id!r}.")
    return TOPOLOGIES[topology_id]


def resolve_topology_and_gate(
    topology_id: Any,
    site_gate: Any,
) -> tuple[Topology, dict[str, Any] | None]:
    topology = resolve_topology(topology_id)
    if topology.topology_id == "A0":
        if site_gate is not None:
            raise ValueError("A0 requires site_gate: null.")
        return topology, None
    if not isinstance(site_gate, Mapping):
        raise ValueError(f"{topology.topology_id} requires an explicit site_gate mapping.")
    extra = set(site_gate) - {"operator", "kappa"}
    if extra:
        raise ValueError("Unsupported site_gate fields: " + ", ".join(sorted(extra)))
    operator = site_gate.get("operator")
    if operator == "relu":
        if "kappa" in site_gate:
            raise ValueError("ReLU does not accept kappa.")
        return topology, {"operator": "relu"}
    if operator not in {"one_sided_threshold", "symmetric_threshold"}:
        raise ValueError("Gate operator must be relu, one_sided_threshold, or symmetric_threshold.")
    if "kappa" not in site_gate:
        raise ValueError(f"{operator} requires kappa.")
    return topology, {"operator": operator, "kappa": _nonnegative_finite(site_gate["kappa"], "kappa")}


def build_gate(spec: Mapping[str, Any], *, torch_module: Any = torch) -> Any:
    operator = spec.get("operator")
    if operator == "relu":
        return torch_module.nn.ReLU()
    if operator == "one_sided_threshold":
        return FixedOneSidedThreshold(spec["kappa"])
    if operator == "symmetric_threshold":
        return FixedSymmetricThreshold(spec["kappa"])
    raise ValueError(f"Unsupported gate operator: {operator!r}.")


def gate_metadata(module: Any) -> dict[str, Any] | None:
    if isinstance(module, torch.nn.ReLU):
        return {"operator": "relu"}
    if isinstance(module, FixedOneSidedThreshold):
        return {"operator": "one_sided_threshold", "kappa": module.kappa}
    if isinstance(module, FixedSymmetricThreshold):
        return {"operator": "symmetric_threshold", "kappa": module.kappa}
    return None


def _nonnegative_finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{label} must be a finite nonnegative number.")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{label} must be a finite nonnegative number.")
    return result


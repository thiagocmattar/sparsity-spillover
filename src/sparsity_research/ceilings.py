"""Analytic topology reach ceilings for full-sequence Pythia product counts."""

from __future__ import annotations

from typing import Any

from .metrics import LOGICAL_OPERATIONS
from .sites import resolve_topology


def architecture_ceiling(
    topology_id: str,
    *,
    layers: int,
    hidden_size: int,
    ffn_size: int,
    sequence_length: int,
    vocabulary_size: int,
) -> dict[str, Any]:
    """Return integer counts and R_model_max for one full uncached sequence."""

    layers = _positive_integer(layers, "layers")
    hidden = _positive_integer(hidden_size, "hidden_size")
    ffn = _positive_integer(ffn_size, "ffn_size")
    tokens = _positive_integer(sequence_length, "sequence_length")
    vocabulary = _positive_integer(vocabulary_size, "vocabulary_size")
    topology = resolve_topology(topology_id)
    active = frozenset(topology.active_sites)

    causal_pairs = tokens * (tokens + 1) // 2
    per_block = {
        "qkv_projection": tokens * 3 * hidden * hidden,
        "qk_scores": hidden * causal_pairs,
        "probability_value": hidden * causal_pairs,
        "attention_output_projection": tokens * hidden * hidden,
        "mlp_w1": tokens * hidden * ffn,
        "mlp_w2": tokens * ffn * hidden,
    }
    if tuple(per_block) != LOGICAL_OPERATIONS:
        raise RuntimeError("Ceiling operation order differs from measured logical operations.")

    reachable = {
        "qkv_projection": "a" in active,
        "qk_scores": bool(active.intersection({"q_pre", "k_pre", "q_post", "k_post"})),
        "probability_value": "v" in active,
        "attention_output_projection": "v" in active,
        "mlp_w1": "m" in active,
        "mlp_w2": "h" in active,
    }
    reachable_per_block = sum(
        per_block[name] for name in LOGICAL_OPERATIONS if reachable[name]
    )
    block_products = sum(per_block.values()) * layers
    reachable_products = reachable_per_block * layers
    lm_head_products = tokens * hidden * vocabulary
    model_products = block_products + lm_head_products
    fraction = reachable_products / model_products
    return {
        "topology_id": topology.topology_id,
        "active_sites": list(topology.active_sites),
        "layers": layers,
        "hidden_size": hidden,
        "ffn_size": ffn,
        "sequence_length": tokens,
        "vocabulary_size": vocabulary,
        "valid_causal_pairs_per_sequence": causal_pairs,
        "per_block_operation_products": per_block,
        "reachable_operations": [
            name for name in LOGICAL_OPERATIONS if reachable[name]
        ],
        "reachable_product_count": reachable_products,
        "block_product_count": block_products,
        "lm_head_product_count": lm_head_products,
        "model_product_count": model_products,
        "R_model_max_fraction": fraction,
        "R_model_max_percent": 100.0 * fraction,
    }


def _positive_integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer.")
    return value

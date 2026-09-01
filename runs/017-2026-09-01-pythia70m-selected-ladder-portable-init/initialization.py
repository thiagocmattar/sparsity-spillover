"""Pinned Pythia-70M recipe initialization mapped onto Transformers GPT-NeoX."""

from __future__ import annotations

import math
from typing import Any


EXPECTED_ARCHITECTURE = {
    "vocab_size": 50304,
    "num_hidden_layers": 6,
    "hidden_size": 512,
    "intermediate_size": 2048,
    "num_attention_heads": 8,
    "max_position_embeddings": 2048,
    "rotary_pct": 0.25,
    "hidden_act": "gelu",
    "layer_norm_eps": 1e-5,
    "use_parallel_residual": True,
    "attention_bias": True,
    "tie_word_embeddings": False,
}
RESIDUAL_OUTPUT_SUFFIXES = (".attention.dense", ".mlp.dense_4h_to_h")


def apply_pythia_70m_initialization(model: Any, *, torch: Any) -> dict[str, Any]:
    """Apply NeoX small_init/Wang distributions without claiming bitwise RNG parity."""

    config = getattr(model, "config", None)
    if config is None:
        raise ValueError("Pythia initialization requires a model config.")
    mismatches = {
        key: (_config_value(config, key), expected)
        for key, expected in EXPECTED_ARCHITECTURE.items()
        if _config_value(config, key) != expected
    }
    if mismatches:
        raise ValueError(f"Pinned Pythia-70M architecture mismatch: {mismatches}")

    layers = int(config.num_hidden_layers)
    hidden = int(config.hidden_size)
    small_std = math.sqrt(2.0 / (5.0 * hidden))
    wang_std = 2.0 / layers / math.sqrt(hidden)
    ordinary_weights = residual_output_weights = biases = normalizations = 0
    with torch.no_grad():
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=small_std)
                ordinary_weights += int(module.weight.numel())
                if name.endswith(RESIDUAL_OUTPUT_SUFFIXES):
                    torch.nn.init.normal_(module.weight, mean=0.0, std=wang_std)
                    ordinary_weights -= int(module.weight.numel())
                    residual_output_weights += int(module.weight.numel())
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
                    biases += int(module.bias.numel())
            elif isinstance(module, torch.nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=small_std)
                ordinary_weights += int(module.weight.numel())
                if module.padding_idx is not None:
                    module.weight[int(module.padding_idx)].zero_()
            elif isinstance(module, torch.nn.LayerNorm) and module.elementwise_affine:
                torch.nn.init.ones_(module.weight)
                normalizations += int(module.weight.numel())
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
                    normalizations += int(module.bias.numel())

    config._attn_implementation = "sdpa"
    config.use_cache = False
    config.pythia_recipe_initialization = {
        "ordinary": "small_init",
        "residual_output": "wang_init",
        "small_init_std": small_std,
        "wang_init_std": wang_std,
        "rng_note": "Transformers module traversal; not GPT-NeoX-bitwise draw order",
    }
    return {
        **config.pythia_recipe_initialization,
        "ordinary_weight_elements": ordinary_weights,
        "residual_output_weight_elements": residual_output_weights,
        "bias_elements_zeroed": biases,
        "normalization_elements_initialized": normalizations,
    }


def verify_recipe_model(model: Any) -> dict[str, Any]:
    config = model.config
    if getattr(config, "_attn_implementation", None) != "sdpa":
        raise ValueError("Training attention must map Pythia flash attention to PyTorch SDPA.")
    if bool(getattr(config, "use_cache", True)):
        raise ValueError("Training must disable the KV cache.")
    if float(getattr(config, "hidden_dropout", math.nan)) != 0.0:
        raise ValueError("Pythia hidden dropout must be zero.")
    if float(getattr(config, "attention_dropout", math.nan)) != 0.0:
        raise ValueError("Pythia attention dropout must be zero.")
    metadata = getattr(config, "pythia_recipe_initialization", None)
    if not isinstance(metadata, dict):
        raise ValueError("Pythia recipe initialization metadata is missing.")
    return {
        "architecture": {key: _config_value(config, key) for key in EXPECTED_ARCHITECTURE},
        "attention_implementation": "sdpa_flash",
        "initialization": dict(metadata),
    }


def _config_value(config: Any, key: str) -> Any:
    if key != "rotary_pct":
        return getattr(config, key, None)
    legacy = getattr(config, "rotary_pct", None)
    if legacy is not None:
        return legacy
    rope = getattr(config, "rope_parameters", None)
    return rope.get("partial_rotary_factor") if isinstance(rope, dict) else None


# Compatibility name required while importing the frozen Run 004 lifecycle. The
# function itself validates and initializes only the pinned 70M architecture.
apply_pythia_14m_initialization = apply_pythia_70m_initialization

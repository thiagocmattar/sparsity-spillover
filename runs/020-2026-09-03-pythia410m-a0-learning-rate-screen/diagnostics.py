"""Run 020 eager diagnostics with exact z-site and endpoint semantics."""

from contextlib import contextmanager
from typing import Any, Mapping

from _reuse_run004 import load_run004_module


_BASE = load_run004_module("_run020_frozen_run004_diagnostics", "diagnostics.py")
AttentionOutputCapture = _BASE.AttentionOutputCapture
logical_product_validation = _BASE.logical_product_validation


@contextmanager
def eager_attention_context(_torch: Any, _device: Any):
    """Force the same eager attention implementation used by logical counters."""

    # The concrete model config is switched by the caller below. This context
    # replaces Run 004's flash-only context without hiding any model mutation.
    yield


def activation_diagnostic_validation(
    *, model: Any, tokens: Any, config: Mapping[str, Any], torch: Any, np: Any
) -> tuple[dict[str, Any], dict[str, Any]]:
    original = model.config._attn_implementation
    model.config._attn_implementation = "eager"
    previous_context = _BASE.recipe_attention_context
    _BASE.recipe_attention_context = eager_attention_context
    try:
        coverage, statistics = _BASE.activation_diagnostic_validation(
            model=model,
            tokens=tokens,
            config=config,
            torch=torch,
            np=np,
        )
    finally:
        _BASE.recipe_attention_context = previous_context
        model.config._attn_implementation = original
    statistics.setdefault("site_definition", {})["z"] = (
        "concatenated PV context immediately before attention.dense (W_o)"
    )
    statistics["attention_implementation"] = "eager"
    return coverage, statistics

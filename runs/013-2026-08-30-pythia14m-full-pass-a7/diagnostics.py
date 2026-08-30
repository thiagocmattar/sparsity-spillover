"""Run 004 FP16 diagnostics with the exact signed Run 013 z-site definition."""

from typing import Any, Mapping

from _reuse_run004 import load_run004_module


_BASE = load_run004_module("_run013_frozen_run004_diagnostics", "diagnostics.py")

AttentionOutputCapture = _BASE.AttentionOutputCapture
logical_product_validation = _BASE.logical_product_validation


def activation_diagnostic_validation(
    *, model: Any, tokens: Any, config: Mapping[str, Any], torch: Any, np: Any
) -> tuple[dict[str, Any], dict[str, Any]]:
    coverage, statistics = _BASE.activation_diagnostic_validation(
        model=model,
        tokens=tokens,
        config=config,
        torch=torch,
        np=np,
    )
    statistics.setdefault("site_definition", {})["z"] = (
        "concatenated PV context immediately before attention.dense (W_o)"
    )
    return coverage, statistics

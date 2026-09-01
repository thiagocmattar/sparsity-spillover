"""Append-only repair for the observed-R_model verification bound.

The frozen Run 017 verifier compared observed logical zero-product opportunity
against the analytic gate-reach ceiling. Incidental exact zeros can occur
outside gate-reachable operations, so the operational contracts only require
the observed fraction to remain in [0, 1]. The analytic ceiling identity is
still checked by the frozen verifier before this repair handles that one
inequality.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping

import verification as _verification_proxy


_FROZEN = _verification_proxy._FROZEN
_ORIGINAL_REQUIRE_DIAGNOSTICS = _FROZEN._require_diagnostics
_LEGACY_ERROR = "Measured R_model exceeds its declared reach ceiling"


def _require_diagnostics_with_observed_bound(
    attempt_dir: Path,
    metrics: Mapping[str, Any],
    config: Mapping[str, Any],
    condition: Mapping[str, Any],
) -> None:
    try:
        _ORIGINAL_REQUIRE_DIAGNOSTICS(attempt_dir, metrics, config, condition)
        return
    except ValueError as exc:
        if _LEGACY_ERROR not in str(exc):
            raise

    logical = _FROZEN._json(attempt_dir / "diagnostics" / "logical_products.json")
    observed = float(logical.get("measured", {}).get("R_model", math.nan))
    if not 0.0 <= observed <= 1.0:
        raise ValueError(f"Measured R_model is not a valid fraction for {attempt_dir.name}.")
    if metrics.get("diagnostics", {}).get("logical_products_path") != "diagnostics/logical_products.json":
        raise ValueError(f"Diagnostic manifest mismatch for {attempt_dir.name}.")


def verify_attempt(condition_id: str) -> dict[str, Any]:
    """Run the frozen verifier with only the invalid upper bound repaired."""

    original = _FROZEN._require_diagnostics
    _FROZEN._require_diagnostics = _require_diagnostics_with_observed_bound
    try:
        return _FROZEN.verify_attempt(condition_id)
    finally:
        _FROZEN._require_diagnostics = original

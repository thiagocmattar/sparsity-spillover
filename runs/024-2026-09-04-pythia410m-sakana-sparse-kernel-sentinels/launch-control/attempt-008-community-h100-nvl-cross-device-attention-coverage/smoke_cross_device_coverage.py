#!/usr/bin/env python3
"""Focused smoke checks for Run-024's cross-device attention accounting."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


def main() -> None:
    repo_root = Path(__file__).resolve().parents[4]
    run_dir = repo_root / "runs" / "024-2026-09-04-pythia410m-sakana-sparse-kernel-sentinels"
    sys.path.insert(0, str(run_dir))
    sys.path.insert(1, str(repo_root / "src"))
    path = run_dir / "03_benchmark.py"
    spec = importlib.util.spec_from_file_location("_run024_coverage_smoke", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    benchmark = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = benchmark
    spec.loader.exec_module(benchmark)

    logical = {
        "measured": {
            "per_operation": {
                "qk_scores": {"zero_product_count": 60},
                "probability_value": {"zero_product_count": 200},
                "mlp_w2": {"zero_product_count": 100},
            },
            "model_product_count": 1000,
            "R_model": 0.5,
            "block_zero_product_count": 500,
        }
    }
    incomparable = benchmark.covered_opportunity(
        logical,
        ["mlp_w2"],
        attention={"q_only_causal_zero_products": 50, "v_only_causal_zero_products": 201},
    )
    assert incomparable["covered_zero_product_count"] is None
    assert incomparable["R_covered"] is None
    assert incomparable["comparable_to_canonical_R_model"] is False
    assert incomparable["cross_device_component_excess"]["v_only_minus_canonical_pv_union"] == 1

    comparable = benchmark.covered_opportunity(
        logical,
        ["mlp_w2"],
        attention={"q_only_causal_zero_products": 50, "v_only_causal_zero_products": 190},
    )
    assert comparable["covered_zero_product_count"] == 340
    assert comparable["R_covered"] == 0.34
    assert comparable["comparable_to_canonical_R_model"] is True
    print("PASS: exact no-cap cross-device coverage and unchanged comparable path")


if __name__ == "__main__":
    main()

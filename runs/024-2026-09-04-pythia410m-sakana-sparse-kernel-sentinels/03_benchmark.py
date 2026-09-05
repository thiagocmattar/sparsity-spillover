#!/usr/bin/env python3
"""Run the frozen Run-023 benchmark with Pythia-410M shape specialization."""

import sys
from typing import Any

from benchmark_core import LINEAR_OPERATION_SPECS, OccupancyAccumulator
from _reuse_run023 import load_run023_module


_impl = load_run023_module("_run023_benchmark", "03_benchmark.py")
_base_load_config = _impl.load_config
INCLUDE_BATCH32 = "--include-batch32" in sys.argv


def make_accumulators(condition: dict, config: dict) -> dict[str, Any]:
    sites = {LINEAR_OPERATION_SPECS[operation]["site"] for operation in condition["linear_operations"]}
    if condition["attention_microbenchmark"]:
        sites.update({"q_post", "k_post", "v"})
    architecture = config["model"]["architecture"]
    widths = {
        "a": architecture["hidden_size"],
        "m": architecture["hidden_size"],
        "h": architecture["intermediate_size"],
        "z": architecture["hidden_size"],
        "q_post": architecture["head_size"],
        "k_post": architecture["head_size"],
        "v": architecture["head_size"],
    }
    result = {}
    for layer in range(architecture["layers"]):
        for site in sorted(sites):
            width = widths[site]
            tile_width = min(config["measurement"]["tile_width"], width)
            capacities = (
                config["measurement"]["tile_payload_capacity"]
                if tile_width == 256
                else {f"factor_{factor}": tile_width // factor - 1 for factor in (8, 4, 2)}
            )
            result[f"{site}.layer_{layer}"] = OccupancyAccumulator(
                width=width,
                tile_width=tile_width,
                near_zero_thresholds=tuple(config["measurement"]["near_zero_thresholds"]),
                payload_capacities=capacities,
            )
    return result


_impl.make_accumulators = make_accumulators


def load_resolved_config() -> dict:
    config = _base_load_config()
    if INCLUDE_BATCH32:
        config["measurement"]["full_model_batches"] = [1, 32]
    return config


_impl.load_config = load_resolved_config


if __name__ == "__main__":
    if INCLUDE_BATCH32:
        sys.argv.remove("--include-batch32")
    _impl.main()

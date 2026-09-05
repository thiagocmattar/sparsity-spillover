"""Run-024 shape specialization over the frozen Run-023 benchmark core."""

from _reuse_run023 import load_run023_module


_impl = load_run023_module("_run023_benchmark_core", "benchmark_core.py")

LINEAR_OPERATION_SPECS = {
    "qkv_projection": {"site": "a", "K": 1024, "N": 3072},
    "mlp_w1": {"site": "m", "K": 1024, "N": 4096},
    "mlp_w2": {"site": "h", "K": 4096, "N": 1024},
    "attention_output_projection": {"site": "z", "K": 1024, "N": 1024},
}

AttentionOpportunityAccumulator = _impl.AttentionOpportunityAccumulator
OccupancyAccumulator = _impl.OccupancyAccumulator
covered_opportunity = _impl.covered_opportunity
pack_exact_ell_reference = _impl.pack_exact_ell_reference
paired_speedup = _impl.paired_speedup
relative_errors = _impl.relative_errors
summarize_latency = _impl.summarize_latency
unpack_exact_ell = _impl.unpack_exact_ell
validate_exact_ell = _impl.validate_exact_ell


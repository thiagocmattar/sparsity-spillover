#!/usr/bin/env python3
"""Fail-closed verification of Run-022 remote artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from run_config import load_config, write_json


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def finite_positive(value: object) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value)) and float(value) > 0


def verify_upstream(path: Path, config: dict) -> dict:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    require(len(rows) == 2, "Upstream CSV must contain exactly TwELL and Torch rows.")
    require([row["implementation"] for row in rows] == ["twell", "torch"], "Unexpected upstream implementation order.")
    expected = config["upstream"]["reference_workload"]
    for row in rows:
        require(int(row["batch_size"]) == expected["batch_size"], "Upstream batch mismatch.")
        require(int(row["seq_len"]) == expected["sequence_length"], "Upstream sequence mismatch.")
        require(row["dtype"] == "bf16" and row["device"] == "cuda", "Upstream dtype/device mismatch.")
        require(int(row["reps"]) == expected["repetitions"], "Upstream repetition mismatch.")
        require(int(row["warmup_reps"]) == expected["warmups"], "Upstream warmup mismatch.")
        require(finite_positive(float(row["avg_total_ms"])), "Invalid upstream latency.")
    latencies = {row["implementation"]: float(row["avg_total_ms"]) for row in rows}
    if expected["require_twell_faster_than_torch"]:
        require(latencies["twell"] < latencies["torch"], "The upstream TwELL positive control did not beat Torch.")
    return {"rows": 2, "latency_ms": latencies, "twell_speedup": latencies["torch"] / latencies["twell"]}


def verify_results(path: Path, config: dict) -> dict:
    result = json.loads(path.read_text(encoding="utf-8"))
    validation = result["validation"]
    source = validation["source_fp32_parameters_fp16_autocast"]
    bf16 = validation["bf16_parameters"]
    expected_validation = config["validation"]
    for record in (source, bf16):
        require(record["complete_block_coverage"] is True, "Validation must use complete blocks.")
        require(record["sequences"] == expected_validation["complete_blocks"], "Validation block count mismatch.")
        require(record["input_tokens"] == expected_validation["evaluated_tokens"], "Validation token count mismatch.")
        require(record["excluded_tail_tokens"] == expected_validation["excluded_tail_tokens"], "Validation tail mismatch.")
        require(finite_positive(record["loss"]), "Validation loss must be finite and positive.")
    require(source["absolute_loss_difference"] <= config["model"]["archived_validation"]["tolerance"], "Archived loss mismatch.")

    occupancy = result["h_occupancy"]
    require(len(occupancy) == config["model"]["architecture"]["layers"], "Expected six layer occupancy rows.")
    expected_elements = expected_validation["evaluated_tokens"] * config["model"]["architecture"]["intermediate_size"]
    for layer, record in enumerate(occupancy):
        require(record["layer"] == layer, "Occupancy layer ordering mismatch.")
        require(record["elements"] == expected_elements, "Occupancy denominator mismatch.")
        require(record["rows"] == expected_validation["evaluated_tokens"], "Occupancy row count mismatch.")
        require(record["integer_pooling"] is True, "Occupancy was not integer pooled.")
        require(0 <= record["exact_zero_count"] <= record["elements"], "Invalid exact-zero count.")
        require(finite_positive(record["rms"]), "Invalid activation RMS.")

    for batch_size in config["measurement"]["full_model_batches"]:
        pair = result["full_model"][str(batch_size)]
        for name in ("dense", "identity_hook"):
            require(len(pair[name]["block_ms"]) == config["measurement"]["full_model_blocks"], "Full-model block count mismatch.")
            require(finite_positive(pair[name]["median_ms"]), "Invalid full-model latency.")

    raw = result["raw_ell"]
    expected_records = config["model"]["architecture"]["layers"] * len(config["measurement"]["raw_ell_rows"])
    require(len(raw) == expected_records, "Raw ELL record count mismatch.")
    tolerance = config["measurement"]["correctness_relative_l2_tolerance"]
    expected_pairs = {
        (layer, rows)
        for layer in range(config["model"]["architecture"]["layers"])
        for rows in config["measurement"]["raw_ell_rows"]
    }
    require({(row["layer"], row["rows"]) for row in raw} == expected_pairs, "Raw ELL coverage mismatch.")
    for record in raw:
        require(record["exact_pack_round_trip"] is True and record["skipped_rows"] == 0, "ELL dropped or skipped values.")
        require(record["overflow_threshold"] >= record["contract"]["max_row_nnz"], "Unsafe ELL overflow threshold.")
        require(record["errors"]["sparse_vs_fp32"]["relative_l2"] <= tolerance, "ELL correctness tolerance exceeded.")
        for timing in record["timings"].values():
            require(len(timing["block_ms"]) == config["measurement"]["raw_ell_blocks"], "Raw timing block count mismatch.")
            require(finite_positive(timing["median_ms"]), "Invalid raw timing.")
    require(result["interpretation_contract"]["attention_not_tested"] is True, "Attention scope was changed.")
    return {
        "source_validation_loss": source["loss"],
        "bf16_validation_loss": bf16["loss"],
        "occupancy_layers": len(occupancy),
        "raw_ell_records": len(raw),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempt-dir", type=Path, required=True)
    args = parser.parse_args()
    attempt = args.attempt_dir.resolve()
    config = load_config()
    static = json.loads((attempt / "static-preflight.json").read_text(encoding="utf-8"))
    require(static.get("passed") is True, "Static preflight did not pass.")
    require(static.get("upstream", {}).get("commit") == config["upstream"]["commit"], "Pinned upstream checkout was not verified.")
    revision = (attempt / "upstream-model-revision.txt").read_text(encoding="utf-8").strip()
    require(revision == config["upstream"]["positive_control_revision"], "Upstream model revision mismatch.")
    require((attempt / "upstream-model-sha256.txt").stat().st_size > 0, "Missing upstream model file inventory.")
    preflight = json.loads((attempt / "remote-preflight.json").read_text(encoding="utf-8"))
    require(preflight.get("passed") is True, "Remote ELL preflight did not pass.")
    require(preflight.get("skipped_rows") == 0 and preflight.get("exact_pack_round_trip") is True, "Unsafe remote preflight.")
    summary = {
        "schema_version": 1,
        "passed": True,
        "upstream": verify_upstream(attempt / "upstream-positive-control.csv", config),
        "pythia": verify_results(attempt / "a0-results.json", config),
        "remote_preflight": {
            "compute_capability": preflight["compute_capability"],
            "operator": preflight["operator"],
        },
    }
    write_json(attempt / "verification.json", summary)
    print(f"PASS: Run 022 artifact verification -> {attempt / 'verification.json'}")


if __name__ == "__main__":
    main()

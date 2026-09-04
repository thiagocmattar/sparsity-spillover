#!/usr/bin/env python3
"""Fail-closed verification of a completed Run-023 phase artifact."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

from benchmark_core import LINEAR_OPERATION_SPECS
from run_config import condition_by_id, conditions_for_phase, load_config, write_json


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def finite_positive(value: object) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value)) and float(value) > 0


def verify_upstream(path: Path, config: dict) -> dict[str, Any]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    require(len(rows) == 2, "Positive-control CSV must contain TwELL and Torch rows.")
    require([row["implementation"] for row in rows] == ["twell", "torch"], "Unexpected positive-control order.")
    expected = config["upstream"]["positive_control"]
    for row in rows:
        require(int(row["batch_size"]) == expected["batch_size"], "Positive-control batch mismatch.")
        require(int(row["seq_len"]) == expected["sequence_length"], "Positive-control sequence mismatch.")
        require(row["dtype"] == "bf16" and row["device"] == "cuda", "Positive-control dtype/device mismatch.")
        require(int(row["reps"]) == expected["repetitions"], "Positive-control repetition mismatch.")
        require(int(row["warmup_reps"]) == expected["warmups"], "Positive-control warmup mismatch.")
        require(finite_positive(float(row["avg_total_ms"])), "Invalid positive-control latency.")
    latencies = {row["implementation"]: float(row["avg_total_ms"]) for row in rows}
    require(
        not expected["require_twell_faster_than_torch"] or latencies["twell"] < latencies["torch"],
        "Official TwELL positive control did not beat Torch.",
    )
    return {"latency_ms": latencies, "twell_speedup": latencies["torch"] / latencies["twell"]}


def verify_latency_record(record: dict, expected_blocks: int, label: str) -> None:
    require(len(record["block_ms"]) == expected_blocks, f"{label}: timing block count mismatch.")
    require(finite_positive(record["median_ms"]), f"{label}: invalid median latency.")
    require(finite_positive(record["tokens_per_second"]), f"{label}: invalid throughput.")


def verify_condition(path: Path, condition: dict, config: dict) -> dict[str, Any]:
    result = json.loads(path.read_text(encoding="utf-8"))
    require(result["derivative_label"] == "Sakana-derived", "Modified path mislabeled.")
    require(result["condition"]["id"] == condition["id"], "Condition identity mismatch.")
    require(result["topology"]["topology_id"] == condition["topology_id"], "Topology mismatch.")
    expected_validation = config["validation"]
    validations = result["validation"]
    for name in (
        "source_fp32_parameters_fp16_autocast",
        "native_dense_bf16",
        "sparse_linear_bf16",
    ):
        record = validations[name]
        require(record["complete_block_coverage"] is True, f"{condition['id']} {name}: incomplete validation.")
        require(record["sequences"] == expected_validation["complete_blocks"], "Validation block mismatch.")
        require(record["input_tokens"] == expected_validation["evaluated_tokens"], "Validation token mismatch.")
        require(record["excluded_tail_tokens"] == expected_validation["excluded_tail_tokens"], "Validation tail mismatch.")
        require(finite_positive(record["loss"]), "Validation loss must be finite and positive.")
    source = validations["source_fp32_parameters_fp16_autocast"]
    require(
        source["batch_size"] == expected_validation["source_reproduction_batch_size"],
        "Source validation batch size no longer matches Run 018.",
    )
    for name in ("native_dense_bf16", "sparse_linear_bf16"):
        require(
            validations[name]["batch_size"] == expected_validation["runtime_equivalence_batch_size"],
            f"{condition['id']} {name}: runtime-equivalence batch size changed.",
        )
    sparse = validations["sparse_linear_bf16"]
    require(
        source["absolute_loss_difference"] <= expected_validation["source_loss_absolute_tolerance"],
        f"{condition['id']}: archived source loss mismatch.",
    )
    require(
        sparse["absolute_loss_difference_from_dense_bf16"]
        <= expected_validation["sparse_vs_dense_loss_absolute_tolerance"],
        f"{condition['id']}: sparse full-validation loss mismatch.",
    )

    coverage = result["kernel_covered_opportunity"]
    extended_coverage = result["kernel_covered_opportunity_with_separate_attention"]
    logical = result["canonical_logical_products"]["measured"]
    require(abs(logical["R_model"] - condition["canonical_r_model"]) < 1e-15, "Canonical R_model changed.")
    require(abs(coverage["canonical_R_model"] - condition["canonical_r_model"]) < 1e-15, "Coverage R_model mismatch.")
    require(
        coverage["covered_zero_product_count"] <= logical["block_zero_product_count"],
        "Covered opportunity exceeds canonical opportunity.",
    )
    require(
        abs(
            coverage["R_covered"]
            - coverage["covered_zero_product_count"] / coverage["canonical_model_product_count"]
        )
        < 1e-15,
        "R_covered was not derived from integers.",
    )

    sites = {LINEAR_OPERATION_SPECS[operation]["site"] for operation in condition["linear_operations"]}
    if condition["attention_microbenchmark"]:
        sites.update({"q_post", "k_post", "v"})
    expected_occupancy = {
        f"{site}.layer_{layer}"
        for site in sites
        for layer in range(config["model"]["architecture"]["layers"])
    }
    occupancy = result["runtime_occupancy"]
    require(set(occupancy) == expected_occupancy, "Runtime occupancy site/layer coverage mismatch.")
    for name, record in occupancy.items():
        site = name.split(".")[0]
        attention_site = site in {"q_post", "k_post", "v"}
        expected_rows = expected_validation["evaluated_tokens"] * (8 if attention_site else 1)
        expected_width = 64 if attention_site else {"h": 2048}.get(site, 512)
        require(record["rows"] == expected_rows, f"{name}: occupancy row denominator mismatch.")
        require(record["elements"] == expected_rows * expected_width, f"{name}: element denominator mismatch.")
        require(record["integer_pooling"] is True, f"{name}: occupancy is not integer pooled.")
        require(0 <= record["exact_zero_count"] <= record["elements"], f"{name}: invalid zeros.")
        require(finite_positive(record["rms"]), f"{name}: invalid RMS.")

    measurement = config["measurement"]
    for batch_size in measurement["full_model_batches"]:
        timing = result["full_model"][str(batch_size)]
        for variant in ("native_dense", "adapter_dense", "sparse_linear"):
            verify_latency_record(
                timing[variant], measurement["full_model_blocks"], f"{condition['id']} B{batch_size} {variant}"
            )
        require(
            set(timing["paired_speedup_vs_native"]) == {"adapter_dense", "sparse_linear"},
            "Missing paired full-model comparisons.",
        )

    primitives = result["linear_primitives"]
    expected_primitives = config["model"]["architecture"]["layers"] * len(condition["linear_operations"])
    require(len(primitives) == expected_primitives, "Linear primitive coverage mismatch.")
    expected_pairs = {
        (layer, operation)
        for layer in range(config["model"]["architecture"]["layers"])
        for operation in condition["linear_operations"]
    }
    require({(row["layer"], row["operation"]) for row in primitives} == expected_pairs, "Linear primitive set mismatch.")
    tolerance = measurement["correctness_relative_l2_tolerance"]
    for row in primitives:
        spec = LINEAR_OPERATION_SPECS[row["operation"]]
        require(
            row["shape"]
            == {"M": measurement["primitive_rows"], "K": spec["K"], "N": spec["N"]},
            "Linear primitive shape changed.",
        )
        require(row["contract"]["signed_round_trip_exact"] is True, "Signed ELL round trip failed.")
        require(row["contract"]["dropped_values"] == 0, "ELL values were dropped.")
        require(row["workspace_allocations"] == 1, "Linear timing allocated inside the steady-state workspace.")
        require(row["errors"]["sparse_vs_fp32"]["relative_l2"] <= tolerance, "Linear primitive error exceeded tolerance.")
        for variant in ("dense_bf16", "pack_only", "kernel_only_plus_bias", "pack_plus_kernel_plus_bias"):
            verify_latency_record(
                row["timings"][variant], measurement["primitive_blocks"], f"{condition['id']} primitive {variant}"
            )

    attention = result["attention_compositions"]
    if condition["attention_microbenchmark"]:
        require(attention is not None and len(attention) == config["model"]["architecture"]["layers"], "A7 attention missing.")
        opportunity = result["attention_opportunity"]
        opportunity_validation = result["attention_opportunity_validation"]
        require(opportunity_validation is not None, "A7 opportunity coverage pass missing.")
        require(opportunity_validation["complete_block_coverage"] is True, "A7 opportunity coverage incomplete.")
        require(
            opportunity_validation["batch_size"] == expected_validation["logical_opportunity_batch_size"],
            "A7 opportunity batch size no longer matches Run 018.",
        )
        require(opportunity_validation["input_tokens"] == expected_validation["evaluated_tokens"], "A7 opportunity tokens mismatch.")
        require(opportunity["integer_pooling"] is True, "Attention opportunity is not integer pooled.")
        require(opportunity["layers"] == config["model"]["architecture"]["layers"], "Attention opportunity layers missing.")
        require(
            opportunity["causal_products_per_operation"]
            == logical["per_operation"]["qk_scores"]["product_count"]
            == logical["per_operation"]["probability_value"]["product_count"],
            "A7 attention opportunity denominator does not match canonical causal products.",
        )
        require(extended_coverage is not None, "A7 extended kernel coverage is missing.")
        require(
            extended_coverage["covered_zero_product_count"]
            == coverage["covered_zero_product_count"]
            + opportunity["q_only_causal_zero_products"]
            + opportunity["v_only_causal_zero_products"],
            "A7 extended coverage does not match separate Q/V compositions.",
        )
        for row in attention:
            require(
                row["shape"] == {"B": 1, "H": 8, "T": 2048, "d": 64},
                "A7 attention composition shape changed.",
            )
            require(row["errors"]["sparse_vs_dense_bf16"]["relative_l2"] <= tolerance, "Attention error exceeded tolerance.")
            require(isinstance(row["break_even"], bool), "Attention break-even decision missing.")
            for variant in ("dense_bf16", "sparse_q_and_v"):
                verify_latency_record(
                    row["composition_timings"][variant],
                    measurement["primitive_blocks"],
                    f"{condition['id']} attention {variant}",
                )
    else:
        require(
            attention is None
            and result["attention_opportunity"] is None
            and result["attention_opportunity_validation"] is None,
            "Non-A7 attention scope changed.",
        )
        require(extended_coverage is None, "Non-A7 extended attention coverage must be absent.")
    contract = result["interpretation_contract"]
    require(contract["r_model_is_logical_not_speedup"] is True, "R_model interpretation changed.")
    require(contract["lm_head_remains_dense"] is True and contract["teal_not_applied"] is True, "Scope changed.")
    return {
        "condition_id": condition["id"],
        "source_loss": source["loss"],
        "dense_bf16_loss": validations["native_dense_bf16"]["loss"],
        "sparse_bf16_loss": sparse["loss"],
        "R_model": condition["canonical_r_model"],
        "R_covered": coverage["R_covered"],
        "R_covered_with_separate_attention": (
            extended_coverage["R_covered"] if extended_coverage is not None else None
        ),
        "full_model_speedup_b1": result["full_model"]["1"]["paired_speedup_vs_native"]["sparse_linear"]["median"],
        "full_model_speedup_b32": result["full_model"]["32"]["paired_speedup_vs_native"]["sparse_linear"]["median"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempt-dir", type=Path, required=True)
    parser.add_argument("--phase", choices=("sentinel", "remainder"), required=True)
    args = parser.parse_args()
    attempt = args.attempt_dir.resolve()
    config = load_config()
    expected_conditions = conditions_for_phase(config, args.phase)
    static = json.loads((attempt / "static-preflight.json").read_text(encoding="utf-8"))
    require(static.get("passed") is True and static["phase"] == args.phase, "Static preflight did not pass this phase.")
    require(static["upstream"]["commit"] == config["upstream"]["commit"], "Upstream commit was not verified.")
    require(static["upstream"]["patch_state"] == "exact_patch_applied", "Exact derivative patch was not applied.")
    official_static = json.loads((attempt / "static-preflight-official.json").read_text(encoding="utf-8"))
    require(official_static.get("passed") is True, "Official static preflight did not pass.")
    require(official_static["upstream"]["commit"] == config["upstream"]["commit"], "Official commit mismatch.")
    require(
        official_static["upstream"]["patch_state"] == "clean_patch_applicable",
        "Positive control was not checked against a clean official tree.",
    )
    revision = (attempt / "upstream-model-revision.txt").read_text(encoding="utf-8").strip()
    require(revision == config["upstream"]["positive_control"]["revision"], "Positive-control model revision mismatch.")
    require((attempt / "upstream-model-sha256.txt").stat().st_size > 0, "Positive-control model inventory missing.")
    preflight = json.loads((attempt / "remote-preflight.json").read_text(encoding="utf-8"))
    require(preflight.get("passed") is True, "Remote derived-kernel preflight did not pass.")
    require(preflight["run022_n128_regression_fixed"] is True, "N=128 regression was not fixed.")
    require(len(preflight["shape_checks"]) == 7, "Remote shape matrix is incomplete.")
    cohort = json.loads((attempt / "cohort.json").read_text(encoding="utf-8"))
    require(cohort["phase"] == args.phase, "Cohort phase mismatch.")
    require(cohort["condition_ids"] == [condition["id"] for condition in expected_conditions], "Cohort order mismatch.")
    require(cohort["completed_conditions"] == len(expected_conditions), "Cohort is incomplete.")
    environment = cohort["environment"]
    require(environment["torch"].split("+")[0] == config["runtime"]["pythia_torch"], "Torch version mismatch.")
    require(environment["transformers"] == config["runtime"]["pythia_transformers"], "Transformers version mismatch.")
    require(environment["numpy"] == config["runtime"]["pythia_numpy"], "NumPy version mismatch.")
    require(environment["compute_capability"] == config["runtime"]["compute_capability"], "GPU capability mismatch.")
    require(environment["upstream_commit"] == config["upstream"]["commit"], "Cohort upstream mismatch.")
    require(environment["patch_sha256"] == config["upstream"]["patch_sha256"], "Cohort patch mismatch.")
    require(environment["gpu"]["identity"] == preflight["gpu"], "Preflight and benchmark GPU identities differ.")
    summaries = [
        verify_condition(attempt / f"{condition['id']}.json", condition, config)
        for condition in expected_conditions
    ]
    attention_rows = [
        row
        for condition in expected_conditions
        if condition["attention_microbenchmark"]
        for row in json.loads((attempt / f"{condition['id']}.json").read_text(encoding="utf-8"))[
            "attention_compositions"
        ]
    ]
    require(
        cohort["attention_promotion"]["any_layer_break_even"]
        == any(row["break_even"] for row in attention_rows),
        "Attention promotion summary mismatch.",
    )
    summary = {
        "schema_version": 1,
        "passed": True,
        "phase": args.phase,
        "upstream_positive_control": verify_upstream(attempt / "upstream-positive-control.csv", config),
        "remote_preflight": {
            "compute_capability": preflight["compute_capability"],
            "shape_checks": len(preflight["shape_checks"]),
            "run022_n128_regression_fixed": True,
        },
        "conditions": summaries,
        "attention_promotion": cohort["attention_promotion"],
    }
    write_json(attempt / "verification.json", summary)
    print(f"PASS: Run 023 {args.phase} verification -> {attempt / 'verification.json'}")


if __name__ == "__main__":
    main()

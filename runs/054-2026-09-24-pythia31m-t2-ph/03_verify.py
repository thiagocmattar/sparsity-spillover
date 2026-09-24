"""Verify one retrieved condition or the complete six-condition training cohort."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from sparsity_research.artifacts import build_transfer_inventory, verify_transfer_inventory
from sparsity_research.ceilings import architecture_ceiling
from run_config import (RUN_DIR, load_config, condition_specs, resolved_condition_config,
    EXPECTED_INITIAL_PARAMETER_SHA256, build_schedule, require_validation_coverage,
    inventory_content_sha256, write_json)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def require(value, message):
    if not value:
        raise ValueError(message)


def verify(attempt, config):
    manifest, metrics = (read(attempt/name) for name in ("manifest.json", "metrics.json"))
    row = next(r for r in condition_specs(config) if r["id"] == manifest["condition"]["id"])
    resolved = resolved_condition_config(config, row)
    require(manifest["status"] == "completed", "Training must be terminally complete")
    require(manifest["condition"] == metrics["condition"] == row, "Condition mismatch")
    require(manifest["model"]["architecture"] == config["model"]["architecture"], "Wrong architecture")
    require(manifest["initial_parameter_sha256"] == EXPECTED_INITIAL_PARAMETER_SHA256, "Initialization mismatch")
    _, order_hash, order = build_schedule(config, {"tokens":1491711416}, np=np)
    require(manifest["training_schedule_hash"] == order_hash, "Training order mismatch")
    require(manifest["activation_pressure"] == resolved["activation_pressure"], "Pressure mismatch")
    t = config["training"]
    require(manifest["completed_steps"] == t["max_steps"], "Incomplete update count")
    require(manifest["input_tokens"] == order["scheduled_blocks"]*2048, "Incomplete token budget")
    require(not manifest["gradient_overflow_steps"], "Skipped optimizer updates")
    require(manifest["topology"]["topology_id"] == row["topology_id"], "Wrong checkpoint topology")
    # Match every source that produced this attempt, independent of later README additions.
    for source in manifest["run_code"]["files"]:
        path = RUN_DIR.parents[1]/source["path"]
        payload = path.read_bytes().replace(b"\r\n", b"\n")
        require(hashlib.sha256(payload).hexdigest() == source["sha256"], "Scientific source drift: "+source["path"])
    events = [json.loads(line) for line in (attempt/"events.jsonl").read_text().splitlines() if line]
    train = [event for event in events if event.get("event") == "train"]
    require([event["step"] for event in train] == list(range(1,t["max_steps"]+1)), "Incomplete boundary history")
    capture_hash = hashlib.sha256("\n".join(f"h.layer_{i}" for i in range(6)).encode()).hexdigest()
    for event in train:
        require(not event["optimizer_step_skipped"] and not event["gradient_overflow"], "Overflowed boundary")
        for key in ("task_loss", "step_wall_seconds", "tokens_per_second", "adamw_gradient_norm_post_clip"):
            require(math.isfinite(event[key]), "Nonfinite boundary metric: "+key)
        require(event["adamw_gradient_norm_post_clip"] <= t["gradient_clip_norm"]+1e-5, "Task clip exceeded")
        if not row["is_control"]:
            require(event["pressure_capture_tensor_count"] == 6 and event["pressure_capture_names_sha256"] == capture_hash, "Pressure capture mismatch")
            require(event["ol1_correction_applied"], "Missing OL1 correction")
            require(event["pressure_to_task_ratio_final"] <= row["step_budget"]+1e-9, "Trust budget exceeded")
            for key in ("task_pressure_gradient_dot", "task_pressure_gradient_cosine", "task_pressure_dot_before",
                        "task_pressure_dot_after", "pressure_to_task_ratio_raw", "trust_scale"):
                require(math.isfinite(event[key]), "Missing/nonfinite OL1 metric: "+key)
    for key in ("step_one", "final", "activation_diagnostic", "logical_product_diagnostic_eager"):
        require_validation_coverage(metrics["validation"][key], config)
        require(math.isfinite(metrics["validation"][key]["loss"]), "Nonfinite validation loss")
    activation = read(attempt/"diagnostics/activation_statistics.json")
    require(len(activation["rows"]) == 6*len(config["diagnostics"]["activation_sites"]), "Incomplete activation coverage")
    require(all(r["nonfinite"] == 0 for r in activation["rows"]), "Nonfinite activations")
    weights = read(attempt/"diagnostics/weight_statistics.json")
    require(bool(weights["rows"]) and all(r["nonfinite"] == 0 for r in weights["rows"]), "Invalid weight diagnostics")
    logical = read(attempt/"diagnostics/logical_products.json")
    ceiling = architecture_ceiling(row["topology_id"], layers=6, hidden_size=256, ffn_size=1024,
        sequence_length=2048, vocabulary_size=50304)
    require(logical["architecture_maximum"] == ceiling, "Architecture ceiling mismatch")
    measured = logical["measured"]
    require(measured["R_model"] == measured["block_zero_product_count"]/measured["model_product_count"], "R_model not count-pooled")
    require(measured["R_block"] == measured["block_zero_product_count"]/measured["block_product_count"], "R_block not count-pooled")
    checkpoints = metrics["checkpoints"]
    require(checkpoints["model_steps"] == config["checkpoints"]["model_steps"], "Checkpoint cadence mismatch")
    for step in config["checkpoints"]["model_steps"]:
        folder = attempt/"checkpoints"/f"step_{step:06d}"
        require((folder/"model.safetensors").is_file(), "Missing retained model")
        if step in config["checkpoints"]["optimizer_steps"]:
            require((folder/"training_state.pt").is_file(), "Missing recovery state")
    inventory = build_transfer_inventory(attempt/"checkpoints")
    require(inventory_content_sha256(inventory) == checkpoints["inventory_content_sha256"], "Checkpoint bytes changed")
    verify_transfer_inventory(attempt, read(attempt/"transfer_inventory.json"))
    return dict(condition=row, attempt=attempt.name, status="verified", final_loss=metrics["validation"]["final"]["loss"],
        R_model=measured["R_model"], R_block=measured["R_block"], ceiling=ceiling,
        final_checkpoint=checkpoints["final"], median_tokens_per_second=metrics["training"]["median_tokens_per_second"],
        checkpoint_bytes=checkpoints["total_bytes"])


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--attempt", type=Path)
    args = p.parse_args()
    config = load_config()
    attempts = [args.attempt] if args.attempt else sorted((RUN_DIR/"artifacts/attempts").glob("*"))
    results = [verify(path, config) for path in attempts if path.is_dir()]
    if not args.attempt:
        require({r["condition"]["id"] for r in results} == {r["id"] for r in condition_specs(config)} and len(results)==6,
                "Complete six-condition cohort required")
    write_json(RUN_DIR/"artifacts"/("verification-"+args.attempt.name+".json" if args.attempt else "verification.json"),
               dict(status="verified", conditions=results))
    print("Verified", len(results), "conditions")

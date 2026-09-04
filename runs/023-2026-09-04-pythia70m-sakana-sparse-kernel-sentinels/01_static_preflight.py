#!/usr/bin/env python3
"""Fail-closed local validation of all Run-023 scientific inputs."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from safetensors import safe_open

from run_config import (
    RUN_DIR,
    checkpoint_file_records,
    checkpoint_path,
    conditions_for_phase,
    display_path,
    load_config,
    logical_products_path,
    repo_path,
    verify_file,
    write_json,
)


def expected_gate(condition: dict) -> tuple[object, object]:
    if condition["step"] == "A0":
        return None, None
    if condition["step"] == "A1-H":
        return {"operator": "relu"}, None
    gates = {}
    for site in condition["active_sites"]:
        operator = "symmetric_threshold" if site in {"q_post", "k_post", "v"} else "one_sided_threshold"
        gates[site] = {"operator": operator, "kappa": condition["gate_threshold"]}
    return None, gates


def validate_condition(config: dict, condition: dict) -> dict:
    checkpoint = checkpoint_path(config, condition)
    files = {
        name: verify_file(checkpoint / name, record["sha256"], record["bytes"])
        for name, record in checkpoint_file_records(config, condition).items()
    }
    logical_path = logical_products_path(config, condition)
    logical_record = verify_file(
        logical_path,
        condition["logical_products"]["sha256"],
        condition["logical_products"]["bytes"],
    )
    architecture = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
    expected = config["model"]["architecture"]
    site_gate, site_gates = expected_gate(condition)
    assertions = {
        "topology_id": architecture.get("topology_id") == condition["topology_id"],
        "site_gate": architecture.get("site_gate") == site_gate,
        "site_gates": architecture.get("site_gates") == site_gates,
        "hidden_size": architecture.get("hidden_size") == expected["hidden_size"],
        "intermediate_size": architecture.get("intermediate_size") == expected["intermediate_size"],
        "layers": architecture.get("num_hidden_layers") == expected["layers"],
        "heads": architecture.get("num_attention_heads") == expected["attention_heads"],
        "sequence_length": architecture.get("max_position_embeddings") == expected["sequence_length"],
        "attention_bias": architecture.get("attention_bias") is True,
        "released_weights_not_loaded": architecture["pythia_recipe_initialization"]["released_weights_loaded"] is False,
    }
    metadata = json.loads((checkpoint / "checkpoint_metadata.json").read_text(encoding="utf-8"))
    assertions["optimizer_step"] = metadata.get("step") == config["source_run"]["optimizer_steps"]
    logical = json.loads(logical_path.read_text(encoding="utf-8"))
    assertions["logical_topology"] = logical["architecture_maximum"]["topology_id"] == condition["topology_id"]
    assertions["logical_coverage"] = (
        logical["coverage"]["sequences"] == config["validation"]["complete_blocks"]
        and logical["coverage"]["input_tokens"] == config["validation"]["evaluated_tokens"]
        and logical["coverage"]["excluded_tail_tokens"] == config["validation"]["excluded_tail_tokens"]
    )
    assertions["canonical_r_model"] = (
        abs(float(logical["measured"]["R_model"]) - condition["canonical_r_model"]) < 1e-15
    )
    if not all(assertions.values()):
        raise ValueError(f"Checkpoint assertions failed for {condition['id']}: {assertions}")

    expected_shapes = {
        "attention.query_key_value.weight": (1536, 512),
        "attention.query_key_value.bias": (1536,),
        "attention.dense.weight": (512, 512),
        "attention.dense.bias": (512,),
        "mlp.dense_h_to_4h.weight": (2048, 512),
        "mlp.dense_h_to_4h.bias": (2048,),
        "mlp.dense_4h_to_h.weight": (512, 2048),
        "mlp.dense_4h_to_h.bias": (512,),
    }
    shapes = {}
    with safe_open(checkpoint / "model.safetensors", framework="pt", device="cpu") as handle:
        keys = set(handle.keys())
        for layer in range(expected["layers"]):
            for suffix, expected_shape in expected_shapes.items():
                key = f"gpt_neox.layers.{layer}.{suffix}"
                if key not in keys:
                    raise ValueError(f"Missing tensor in {condition['id']}: {key}")
                actual = tuple(handle.get_slice(key).get_shape())
                if actual != expected_shape:
                    raise ValueError(f"Shape mismatch for {key}: {actual} != {expected_shape}")
                shapes[key] = list(actual)
    return {
        "condition_id": condition["id"],
        "checkpoint": display_path(checkpoint),
        "files": files,
        "logical_products": logical_record,
        "assertions": assertions,
        "linear_tensor_shapes": shapes,
    }


def validate_validation(config: dict) -> dict:
    section = config["validation"]
    metadata_path = repo_path(section["metadata"])
    tokens_path = repo_path(section["tokens"])
    files = {
        "metadata": verify_file(metadata_path, section["metadata_sha256"]),
        "tokens": verify_file(tokens_path, section["tokens_sha256"]),
    }
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assertions = {
        "documents": metadata["documents"] == section["documents"],
        "source_tokens": metadata["tokens"] == section["source_tokens"],
        "complete_blocks": metadata["complete_blocks"] == section["complete_blocks"],
        "evaluated_tokens": metadata["evaluated_complete_block_tokens"] == section["evaluated_tokens"],
        "excluded_tail": metadata["excluded_tail_tokens"] == section["excluded_tail_tokens"],
        "nested_token_hash": metadata["tokens_sha256"] == section["tokens_sha256"],
        "int32_bytes": tokens_path.stat().st_size == section["source_tokens"] * 4,
    }
    if not all(assertions.values()):
        raise ValueError(f"Validation assertions failed: {assertions}")
    return {"files": files, "assertions": assertions}


def _run_git(upstream_dir: Path, arguments: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(upstream_dir), *arguments],
        check=check,
        capture_output=True,
        text=True,
    )


def validate_upstream(config: dict, upstream_dir: Path) -> dict:
    upstream_dir = upstream_dir.resolve()
    commit = _run_git(upstream_dir, ["rev-parse", "HEAD"]).stdout.strip()
    if commit != config["upstream"]["commit"]:
        raise ValueError(f"Upstream commit mismatch: {commit}")
    blobs = {}
    for relative, expected_blob in config["upstream"]["blob_ids"].items():
        actual = _run_git(upstream_dir, ["rev-parse", f"HEAD:{relative}"]).stdout.strip()
        if actual != expected_blob:
            raise ValueError(f"Upstream blob mismatch for {relative}: {actual}")
        blobs[relative] = actual
    patch_path = repo_path(config["upstream"]["patch"])
    verify_file(patch_path, config["upstream"]["patch_sha256"])
    forward = _run_git(
        upstream_dir,
        ["apply", "--check", "--whitespace=nowarn", str(patch_path)],
        check=False,
    )
    reverse = _run_git(
        upstream_dir,
        ["apply", "--reverse", "--check", "--whitespace=nowarn", str(patch_path)],
        check=False,
    )
    if forward.returncode == 0 and reverse.returncode != 0:
        state = "clean_patch_applicable"
    elif reverse.returncode == 0 and forward.returncode != 0:
        state = "exact_patch_applied"
    else:
        raise ValueError(
            "Upstream tree is neither a clean patch target nor the exact patched state: "
            f"forward={forward.stderr!r}, reverse={reverse.stderr!r}"
        )
    changed = [
        line for line in _run_git(upstream_dir, ["diff", "--name-only"]).stdout.splitlines() if line
    ]
    expected_changed = sorted(config["upstream"]["blob_ids"])[:3]
    if state == "clean_patch_applicable" and changed:
        raise ValueError(f"Clean upstream target has unexpected changes: {changed}")
    if state == "exact_patch_applied" and sorted(changed) != expected_changed:
        raise ValueError(f"Patched upstream changed-file set is wrong: {changed}")
    return {
        "directory": str(upstream_dir),
        "commit": commit,
        "blob_ids": blobs,
        "patch_state": state,
        "changed_files": changed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("all", "sentinel", "remainder"), default="all")
    parser.add_argument("--upstream-dir", type=Path)
    parser.add_argument("--output", type=Path, default=RUN_DIR / "prelaunch" / "static-preflight.json")
    args = parser.parse_args()
    config = load_config()
    conditions = config["conditions"] if args.phase == "all" else conditions_for_phase(config, args.phase)
    result = {
        "schema_version": 1,
        "phase": args.phase,
        "conditions": [validate_condition(config, condition) for condition in conditions],
        "validation": validate_validation(config),
        "upstream": validate_upstream(config, args.upstream_dir) if args.upstream_dir else None,
        "passed": True,
    }
    write_json(args.output, result)
    print(f"PASS: static preflight ({len(conditions)} conditions) -> {args.output}")


if __name__ == "__main__":
    main()

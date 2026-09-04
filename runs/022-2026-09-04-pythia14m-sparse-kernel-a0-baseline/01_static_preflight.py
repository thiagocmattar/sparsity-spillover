#!/usr/bin/env python3
"""Validate immutable local inputs and optionally the pinned upstream checkout."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from safetensors import safe_open

from run_config import RUN_DIR, display_path, load_config, repo_path, verify_file, write_json


def validate_checkpoint(config: dict) -> dict:
    model = config["model"]
    checkpoint = repo_path(model["checkpoint"])
    verified = {
        name: verify_file(checkpoint / name, record["sha256"], record["bytes"])
        for name, record in model["files"].items()
    }
    architecture = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
    expected = model["architecture"]
    assertions = {
        "topology_id": architecture.get("topology_id") == "A0",
        "site_gate_absent": architecture.get("site_gate") is None,
        "hidden_act": architecture.get("hidden_act") == "gelu",
        "hidden_size": architecture.get("hidden_size") == expected["hidden_size"],
        "intermediate_size": architecture.get("intermediate_size") == expected["intermediate_size"],
        "layers": architecture.get("num_hidden_layers") == expected["layers"],
        "heads": architecture.get("num_attention_heads") == expected["attention_heads"],
        "attention_bias": architecture.get("attention_bias") is True,
        "sequence_length": architecture.get("max_position_embeddings") == expected["sequence_length"],
    }
    metadata = json.loads((checkpoint / "checkpoint_metadata.json").read_text(encoding="utf-8"))
    assertions["optimizer_step"] = metadata.get("step") == model["optimizer_steps"]
    if not all(assertions.values()):
        raise ValueError(f"Checkpoint config assertions failed: {assertions}")
    expected_shapes = {
        "dense_h_to_4h.weight": (512, 128),
        "dense_h_to_4h.bias": (512,),
        "dense_4h_to_h.weight": (128, 512),
        "dense_4h_to_h.bias": (128,),
    }
    shapes: dict[str, list[int]] = {}
    with safe_open(checkpoint / "model.safetensors", framework="pt", device="cpu") as handle:
        keys = set(handle.keys())
        for layer in range(expected["layers"]):
            for suffix, shape in expected_shapes.items():
                key = f"gpt_neox.layers.{layer}.mlp.{suffix}"
                if key not in keys:
                    raise ValueError(f"Missing checkpoint tensor: {key}")
                actual = tuple(handle.get_slice(key).get_shape())
                if actual != shape:
                    raise ValueError(f"Shape mismatch for {key}: {actual} != {shape}")
                shapes[key] = list(actual)
    return {"directory": display_path(checkpoint), "files": verified, "assertions": assertions, "mlp_shapes": shapes}


def validate_validation(config: dict) -> dict:
    section = config["validation"]
    metadata_path = repo_path(section["metadata"])
    tokens_path = repo_path(section["tokens"])
    verified = {
        "metadata": verify_file(metadata_path, section["metadata_sha256"]),
        "tokens": verify_file(tokens_path, section["tokens_sha256"]),
    }
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    checks = {
        "documents": metadata["documents"] == section["documents"],
        "source_tokens": metadata["tokens"] == section["source_tokens"],
        "complete_blocks": metadata["complete_blocks"] == section["complete_blocks"],
        "evaluated_tokens": metadata["evaluated_complete_block_tokens"] == section["evaluated_tokens"],
        "excluded_tail_tokens": metadata["excluded_tail_tokens"] == section["excluded_tail_tokens"],
        "token_sha_nested": metadata["tokens_sha256"] == section["tokens_sha256"],
        "token_bytes": tokens_path.stat().st_size == section["source_tokens"] * 4,
    }
    if not all(checks.values()):
        raise ValueError(f"Validation cache assertions failed: {checks}")
    return {"files": verified, "assertions": checks}


def validate_upstream(config: dict, upstream_dir: Path) -> dict:
    upstream_dir = upstream_dir.resolve()
    commit = subprocess.check_output(
        ["git", "-C", str(upstream_dir), "rev-parse", "HEAD"], text=True
    ).strip()
    if commit != config["upstream"]["commit"]:
        raise ValueError(f"Upstream commit mismatch: {commit}")
    files = {}
    for relative, digest in config["upstream"]["source_sha256"].items():
        record = verify_file(upstream_dir / relative, digest)
        record["path"] = relative
        files[relative] = record
    return {"directory_role": "external_upstream_checkout", "commit": commit, "files": files}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream-dir", type=Path)
    parser.add_argument("--output", type=Path, default=RUN_DIR / "prelaunch" / "static-preflight.json")
    args = parser.parse_args()
    config = load_config()
    result = {
        "schema_version": 1,
        "checkpoint": validate_checkpoint(config),
        "validation": validate_validation(config),
        "upstream": None,
        "passed": True,
    }
    if args.upstream_dir:
        result["upstream"] = validate_upstream(config, args.upstream_dir)
    write_json(args.output, result)
    print(f"PASS: static preflight -> {args.output}")


if __name__ == "__main__":
    main()

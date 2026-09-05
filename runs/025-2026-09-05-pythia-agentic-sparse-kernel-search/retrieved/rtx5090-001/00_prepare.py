"""Pin all selected weights and extract only the approved development blocks."""
from __future__ import annotations

import argparse
import hashlib
import subprocess

import numpy as np

from run025_common import ROOT, RUN, config, inside, read_json, record, sha256, write_json

SOURCES = (
    ("14m", "004-2026-08-29-pythia14m-full-pass-l1n"),
    ("14m", "014-2026-08-31-pythia14m-full-pass-a7-ol1"),
    ("14m", "015-2026-08-31-pythia14m-corrected-a4-ol1"),
    ("70m", "018-2026-09-01-pythia70m-selected-ladder-canonical-init"),
    ("410m", "019-2026-09-01-pythia410m-selected-ladder-canonical-init"),
)


def canonical_condition(condition):
    name = condition["id"]
    if name in {"gelu-control", "a0-gelu"}:
        return "a0", "A0", None
    if name in {"relu-control", "a1h-relu"}:
        return "a1h", "A1-H", None
    for prefix, family, short in (("a4z-ol1-kappa-", "A4-OL1", "a4"),
                                  ("a4-ol1-kappa-", "A4-OL1", "a4"),
                                  ("a7-ol1-kappa-", "A7-OL1", "a7")):
        if name.startswith(prefix):
            kappa = float(condition["gate_threshold"])
            return f"{short}-{format(kappa, 'g').replace('.', 'p')}", family, kappa
    return None


def discover():
    rows = []
    for size, source in SOURCES:
        for attempt in sorted((ROOT / "runs" / source / "artifacts" / "attempts").iterdir()):
            manifest_path = attempt / "manifest.json"
            if not manifest_path.is_file():
                continue
            manifest = read_json(manifest_path)
            identity = canonical_condition(manifest["condition"])
            if identity is None:
                continue
            name, family, kappa = identity
            if manifest["status"] != "completed" or manifest["completed_steps"] != 712:
                raise ValueError(f"Selected checkpoint is incomplete: {attempt}")
            checkpoint = inside(attempt, manifest["checkpoints"]["final"]["path"])
            architecture = read_json(checkpoint / "config.json")
            logical_path = attempt / "diagnostics" / "logical_products.json"
            logical = read_json(logical_path)
            if logical["coverage"]["sequences"] != 338:
                raise ValueError(f"Canonical counts lack complete validation: {attempt}")
            rows.append({
                "id": f"{size}/{name}", "size": size, "family": family, "kappa": kappa,
                "source_condition": manifest["condition"],
                "partition": "development" if kappa in {None, 0, 0.5} else "untuned",
                "checkpoint": checkpoint.relative_to(ROOT).as_posix(),
                "architecture": {key: architecture[key] for key in
                                 ("hidden_size", "intermediate_size", "num_hidden_layers", "num_attention_heads", "vocab_size")},
                "topology_id": architecture.get("topology_id", "A0"),
                "files": [record(checkpoint / name) for name in
                          ("config.json", "model.safetensors", "checkpoint_metadata.json", "generation_config.json")],
                "provenance": [record(manifest_path), record(attempt / "config.yaml"), record(logical_path)],
                "canonical_logical_products": logical,
            })
            print(f"Pinned {size}/{name}", flush=True)
    expected = {f"{size}/{name}" for size in ("14m", "70m", "410m") for name in
                ("a0", "a1h", *(f"{family}-{k}" for family in ("a4", "a7") for k in ("0", "0p01", "0p05", "0p1", "0p5")))}
    if len(rows) != 36 or {row["id"] for row in rows} != expected:
        raise ValueError("Selected checkpoint inventory does not match the approved 36")
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", required=True, help="Existing clone; reads pinned Git objects, never the dirty working tree")
    args = parser.parse_args()
    cfg = config()
    upstream = subprocess.check_output(["git", "-C", args.upstream, "show",
        cfg["upstream_commit"] + ":custom_models/twell_modules/matmul_t2d.cu"])
    if hashlib.sha256(upstream).hexdigest() != cfg["upstream_t2d_sha256"]:
        raise ValueError("Official upstream T2D source hash mismatch")
    if sha256(RUN / "upstream/matmul_t2d.cu") != cfg["upstream_t2d_sha256"]:
        raise ValueError("Vendored source differs from canonical Git bytes (LF required)")
    cache = ROOT / cfg["inputs"]["cache_root"]
    train, validation = cache / "train/tokens.int32.bin", cache / "validation/tokens.int32.bin"
    for path, expected in ((train, cfg["inputs"]["train_sha256"]), (validation, cfg["inputs"]["validation_sha256"])):
        if sha256(path) != expected:
            raise ValueError(f"Cache hash mismatch: {path}")
    tokens = np.memmap(train, dtype=np.int32, mode="r")
    length = cfg["inputs"]["sequence_length"]
    ids = np.random.default_rng(cfg["inputs"]["development_seed"]).choice(len(tokens) // length,
                     size=cfg["inputs"]["development_blocks"], replace=False).tolist()
    data = np.stack([tokens[index * length:(index + 1) * length] for index in ids])
    RUN.joinpath("prelaunch").mkdir(exist_ok=True)
    data.tofile(RUN / "prelaunch/development.int32.bin")
    result = {"schema_version": 1, "config_sha256": sha256(RUN / "config.json"),
              "checkpoints": discover(), "validation": record(validation),
              "validation_metadata": record(cache / "validation/metadata.json"),
              "development": record(RUN / "prelaunch/development.int32.bin"),
              "development_source_block_ids": ids, "train_cache_sha256": cfg["inputs"]["train_sha256"],
              "upstream_commit": cfg["upstream_commit"], "upstream_t2d_sha256": hashlib.sha256(upstream).hexdigest()}
    write_json(RUN / "prelaunch/input_manifest.json", result)
    print("PASS: 36 model-only identities and train-only development inputs pinned; no GPU used.")


if __name__ == "__main__":
    main()

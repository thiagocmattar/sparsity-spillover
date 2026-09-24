"""Run055 configuration and inherited cache/order identities."""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import random

import yaml
from sparsity_research.data import build_training_schedule
from sparsity_research.pressure import parse_pressure_config
from sparsity_research.sites import resolve_topology_and_gates
from _reuse_run004 import load_run004_module
from run055_topology import register

register()
RUN_DIR = Path(__file__).resolve().parent
REPO_ROOT = RUN_DIR.parents[1]
DEFAULT_CONFIG = RUN_DIR / "config.yaml"
EXPECTED_THRESHOLDS = (0., .01, .05, .1, .5)
EXPECTED_PRESSURE_SITES = ("h",)
INITIALIZATION_METADATA = RUN_DIR / "prelaunch/initialization/metadata.json"
EXPECTED_INITIAL_PARAMETER_SHA256 = (
    json.loads(INITIALIZATION_METADATA.read_text())["parameter_sha256"]
    if INITIALIZATION_METADATA.exists() else None
)
BASE = load_run004_module("_run055_run004_config", "run_config.py")
BASE.RUN_DIR, BASE.REPO_ROOT, BASE.DEFAULT_CONFIG = RUN_DIR, REPO_ROOT, DEFAULT_CONFIG
mapping = BASE.mapping
repo_path = BASE.repo_path
write_json = BASE.write_json
load_verified_caches = BASE.load_verified_caches
cache_identity = BASE.cache_identity
require_training_cache = BASE.require_training_cache
require_validation_coverage = BASE.require_validation_coverage
require_cuda = BASE.require_cuda
parameter_sha256 = BASE.parameter_sha256
inventory_content_sha256 = BASE.inventory_content_sha256


def git_identity():
    deployment = REPO_ROOT/"deployment.json"
    if deployment.exists():
        return json.loads(deployment.read_text())["repository"]
    return BASE.git_identity()


def load_config(path=DEFAULT_CONFIG):
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    validate_config(config)
    return config


def condition_specs(config):
    rows = []
    if config["conditions"]["include_base"]:
        rows.append(dict(id="base", order=1, topology_id="A0", gate_threshold=None,
                         pressure_method="none", pressure_sites=[], pressure_weight=0.,
                         step_budget=None, active_sites=[], label="Base", is_control=True))
    for kappa in config["conditions"]["gate_thresholds"]:
        token = f"{kappa:g}".replace(".", "p")
        rows.append(dict(id=f"a7-h-ol1-kappa-{token}", order=len(rows)+1,
                         topology_id="A7-Z-POST", gate_threshold=float(kappa), active_sites=list(config["conditions"]["active_sites"]),
                         pressure_method="orthogonal_l1", pressure_sites=["h"],
                         pressure_weight=config["conditions"]["pressure_weight"],
                         step_budget=config["conditions"]["step_budget"],
                         label=f"T7/Ph kappa={kappa:g}", is_control=False))
    return rows


def resolved_condition_config(config, condition):
    value = deepcopy(config)
    value["condition"] = deepcopy(condition)
    model = value["model"]
    model.update(topology_id=condition["topology_id"], site_gate=None,
                 pressure_sites=condition["pressure_sites"])
    model["site_gates"] = None if condition["is_control"] else {
        s: {"operator": "symmetric_threshold" if s in config["conditions"]["symmetric_sites"] else "one_sided_threshold",
            "kappa": condition["gate_threshold"]}
        for s in condition["active_sites"]}
    resolve_topology_and_gates(model["topology_id"], None, model["site_gates"])
    value["activation_pressure"] = dict(method=condition["pressure_method"],
        sites=condition["pressure_sites"], weight=condition["pressure_weight"],
        step_budget=condition["step_budget"], eps=1e-12)
    parse_pressure_config(value["activation_pressure"])
    return value


def validate_config(config):
    if not isinstance(config, dict):
        raise ValueError("Config must be a mapping")
    model, t, c = (mapping(config, k) for k in ("model", "training", "conditions"))
    if model.get("initialization") != "random" or model.get("released_weights_loaded") is not False:
        raise ValueError("Random pretraining without released weights is required")
    if c.get("pressure_method") != "orthogonal_l1" or c.get("pressure_sites") != ["h"]:
        raise ValueError("T7/Ph pressure must be orthogonal_l1 at h")
    thresholds = c["gate_thresholds"]
    if not thresholds or len(set(thresholds)) != len(thresholds) or any(
        isinstance(k, bool) or not isinstance(k, (int, float)) or not math.isfinite(k) or k < 0
        for k in thresholds):
        raise ValueError("Thresholds must be unique, finite and nonnegative")
    for key in ("max_steps", "micro_batch_size", "gradient_accumulation_steps", "global_batch_size"):
        if type(t.get(key)) is not int or t[key] <= 0:
            raise ValueError(f"Invalid {key}")
    if t["micro_batch_size"] * t["gradient_accumulation_steps"] != t["global_batch_size"]:
        raise ValueError("Batch decomposition mismatch")
    for key in ("peak_learning_rate", "minimum_learning_rate", "weight_decay", "warmup_fraction"):
        if not math.isfinite(t[key]) or t[key] < 0:
            raise ValueError(f"Invalid {key}")
    if t["minimum_learning_rate"] > t["peak_learning_rate"] or t["warmup_fraction"] > 1:
        raise ValueError("Invalid learning-rate schedule")
    if t["fp16_ol1_overflow_policy"] != "skip_entire_boundary":
        raise ValueError("OL1 requires atomic overflow handling")
    v = config["validation"]
    if tuple(v[k] for k in ("documents", "complete_sequences", "input_tokens", "excluded_tail_tokens")) != (500,338,692224,1444):
        raise ValueError("Complete MiniPile validation is required")
    if config["diagnostics"]["clipping_frontier"] is not None:
        raise ValueError("Post-hoc clipping is outside this design")
    cp = config["checkpoints"]
    if not cp["retain_final"] or t["max_steps"] not in cp["model_steps"]:
        raise ValueError("Final checkpoint must be retained")
    if any(type(s) is not int or s < 0 or s > t["max_steps"] for s in cp["model_steps"]):
        raise ValueError("Invalid checkpoint step")
    if not set(cp["optimizer_steps"]).issubset(cp["model_steps"]):
        raise ValueError("Recovery checkpoints require model checkpoints")
    for row in condition_specs(config):
        resolved_condition_config(config, row)


def worker_conditions(config, worker_id):
    rows = {r["id"]: r for r in condition_specs(config)}
    return [deepcopy(rows[cid]) for cid in config["runpod"]["worker_assignments"][worker_id]]


def build_schedule(config, train_metadata, *, np):
    t = config["training"]
    return build_training_schedule(np, token_count=train_metadata["tokens"],
        block_size=config["data"]["sequence_length"], max_steps=t["max_steps"],
        gradient_accumulation_steps=t["gradient_accumulation_steps"],
        micro_batch_size=t["micro_batch_size"], seed=config["seeds"]["data_order"])


def microbatches_for_step(tokens, step_starts, *, block_size, device, torch, np):
    return (torch.as_tensor(np.stack([tokens[int(s):int(s)+block_size] for s in batch]),
                           dtype=torch.long, device=device) for batch in step_starts)


def seed_everything(torch, seed):
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    BASE.seed_everything(torch, seed)


def scientific_source_paths():
    paths = [*RUN_DIR.glob("*.py"), DEFAULT_CONFIG, RUN_DIR/"architecture_config.json"]
    paths += list((REPO_ROOT/"src/sparsity_research").glob("*.py"))
    old = RUN_DIR.parent/"004-2026-08-29-pythia14m-full-pass-l1n"
    paths += [old/name for name in ("run_config.py", "training.py", "optimizer_boundary.py", "diagnostics.py", "06_build_cache_from_hf.py")]
    paths += [RUN_DIR.parent/"043-2026-09-20-pythia70m-hz-h-only-ol1/optimizer_boundary.py"]
    if INITIALIZATION_METADATA.exists():
        paths.append(INITIALIZATION_METADATA)
    return sorted(set(paths))


def run_code_identity():
    rows = []
    for path in scientific_source_paths():
        payload = path.read_bytes().replace(b"\r\n", b"\n")
        rows.append(dict(path=path.relative_to(REPO_ROOT).as_posix(), bytes=len(payload),
                         sha256=hashlib.sha256(payload).hexdigest()))
    digest = hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()
    return dict(files=rows, content_sha256=digest, text_normalization="CRLF to LF")


validate_science_config = validate_config

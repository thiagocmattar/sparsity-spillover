"""Approved Run 041 HZ plus h-only OL1 full-pass identities."""

from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path
from typing import Any, Mapping

import yaml

from sparsity_research.artifacts import config_sha256
from sparsity_research.pressure import parse_pressure_config
from sparsity_research.sites import resolve_topology_and_gates

from _reuse_run004 import load_run004_module
from run041_topology import register
register()


RUN_DIR = Path(__file__).resolve().parent
REPO_ROOT = RUN_DIR.parents[1]
DEFAULT_CONFIG = RUN_DIR / "config.yaml"
RUN013_DIR = RUN_DIR.parent / "013-2026-08-30-pythia14m-full-pass-a7"
RUN013_VERIFICATION = RUN013_DIR / "artifacts" / "verification.json"
EXPECTED_THRESHOLDS = (0.0, 0.01, 0.05, 0.1)
EXPECTED_ACTIVE_SITES = ("h", "z")
EXPECTED_ONE_SIDED_SITES = ("h", "z")
EXPECTED_SYMMETRIC_SITES = ()
EXPECTED_PRESSURE_SITES = ("h",)
EXPECTED_PRESSURE_CAPTURE_TENSOR_COUNT = 6
EXPECTED_PRESSURE_CAPTURE_NAMES = tuple(
    sorted(
        f"{site}.layer_{layer}"
        for site in EXPECTED_PRESSURE_SITES
        for layer in range(6)
    )
)
EXPECTED_PRESSURE_CAPTURE_NAMES_SHA256 = hashlib.sha256(
    "\n".join(EXPECTED_PRESSURE_CAPTURE_NAMES).encode("utf-8")
).hexdigest()
EXPECTED_CONDITION_IDS = (
    "hz-h-ol1-kappa-0",
    "hz-h-ol1-kappa-0p01",
    "hz-h-ol1-kappa-0p05",
    "hz-h-ol1-kappa-0p1",
)
EXPECTED_RUN013_CONDITION_IDS = (
    "a7-z-post-mixed-kappa-0",
    "a7-z-post-mixed-kappa-0p01",
    "a7-z-post-mixed-kappa-0p05",
    "a7-z-post-mixed-kappa-0p1",
    "a7-z-post-mixed-kappa-0p5",
)
EXPECTED_WORKERS = {condition_id: (condition_id,) for condition_id in EXPECTED_CONDITION_IDS}
EXPECTED_GPU_TYPES = {
    condition_id: "NVIDIA H100 80GB HBM3" for condition_id in EXPECTED_CONDITION_IDS
}
EXPECTED_INITIAL_PARAMETER_SHA256 = (
    "ece58512e94ee2f97d17278fe8af4c1abef9c5f7f9dbdd4087e36d7f67d7af57"
)
EXPECTED_SCHEDULE_SHA256 = (
    "f1755812b4f70806bd137ee900c9338f64c4c2074b6dd8b7661e6bde9b141faa"
)
EXPECTED_RUN013_CODE_SHA256 = (
    "518199e6f845fd88281525f5cdc1e7c323168504d4bf4393593d9c7ddc8d7453"
)
EXPECTED_CEILING_NUMERATOR = 1_006_632_960
EXPECTED_CEILING_DENOMINATOR = 18_825_609_216
EXPECTED_CEILING_FRACTION = EXPECTED_CEILING_NUMERATOR / EXPECTED_CEILING_DENOMINATOR
EXPECTED_MODEL_CHECKPOINTS = (0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 712)
EXPECTED_OPTIMIZER_CHECKPOINTS = (256, 512, 712)


_BASE = load_run004_module("_run041_frozen_run004_config", "run_config.py")
_BASE.RUN_DIR = RUN_DIR
_BASE.REPO_ROOT = REPO_ROOT
_BASE.DEFAULT_CONFIG = DEFAULT_CONFIG


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Run config must be a mapping.")
    validate_config(config)
    return config


def site_gates(kappa: float) -> dict[str, dict[str, Any]]:
    value = float(kappa)
    one_sided = {"operator": "one_sided_threshold", "kappa": value}
    symmetric = {"operator": "symmetric_threshold", "kappa": value}
    return {
        site: dict(one_sided if site in EXPECTED_ONE_SIDED_SITES else symmetric)
        for site in EXPECTED_ACTIVE_SITES
    }


def validate_config(config: Mapping[str, Any]) -> None:
    """Check types, ranges, exact ports and the pretraining invariants."""
    import math
    model=mapping(config,"model")
    if model.get("initialization") != "random": raise ValueError("Random pretraining required")
    topology,gates=resolve_topology_and_gates(model.get("topology_id"),model.get("site_gate"),model.get("site_gates"))
    if topology.active_sites != EXPECTED_ACTIVE_SITES: raise ValueError("Only h/z gates are supported by this run")
    conditions=mapping(config,"conditions")
    if tuple(conditions.get("pressure_sites",())) != EXPECTED_PRESSURE_SITES: raise ValueError("Pressure must target h only")
    if conditions.get("pressure_method") != "orthogonal_l1": raise ValueError("OL1 required")
    thresholds=conditions.get("gate_thresholds",[])
    if not thresholds or len(thresholds)!=len(set(thresholds)) or any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or v<0 for v in thresholds): raise ValueError("Unique finite nonnegative thresholds required")
    for row in condition_specs(config): resolved_condition_config(config,row)
    t=mapping(config,"training")
    for key in ("max_steps","micro_batch_size","gradient_accumulation_steps","global_batch_size"):
        if isinstance(t.get(key),bool) or not isinstance(t.get(key),int) or t[key]<=0: raise ValueError(f"Invalid {key}")
    if t["micro_batch_size"]*t["gradient_accumulation_steps"]!=t["global_batch_size"]: raise ValueError("Batch decomposition mismatch")
    if t.get("fp16_ol1_overflow_policy")!="skip_entire_boundary": raise ValueError("Atomic overflow policy required")
    if any(not math.isfinite(float(t[k])) or float(t[k])<0 for k in ("peak_learning_rate","minimum_learning_rate","weight_decay","warmup_fraction")): raise ValueError("Invalid optimizer scalar")
    if not 0<=float(t["warmup_fraction"])<=1 or t["minimum_learning_rate"]>t["peak_learning_rate"]: raise ValueError("Invalid LR schedule")
    v=mapping(config,"validation")
    if (v.get("documents"),v.get("complete_sequences"),v.get("input_tokens"),v.get("excluded_tail_tokens"))!=(500,338,692224,1444): raise ValueError("Full validation required")
    d=mapping(config,"diagnostics")
    if d.get("clipping_frontier") is not None: raise ValueError("No post-hoc clipping authorized")
    if not d.get("gradient_interaction") or not d.get("ol1_adaptive_directions"): raise ValueError("Training gradient geometry required")
    if not mapping(config,"checkpoints").get("retain_final"): raise ValueError("Final checkpoint retention required")


def condition_specs(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    thresholds = [float(value) for value in mapping(config, "conditions")["gate_thresholds"]]
    return [_condition(order, threshold) for order, threshold in enumerate(thresholds, 1)]


def _condition(order: int, threshold: float) -> dict[str, Any]:
    token = f"{float(threshold):g}".replace(".", "p")
    return {
        "id": f"hz-h-ol1-kappa-{token}",
        "order": int(order),
        "topology_id": "HZ",
        "active_sites": list(EXPECTED_ACTIVE_SITES),
        "one_sided_sites": list(EXPECTED_ONE_SIDED_SITES),
        "symmetric_sites": list(EXPECTED_SYMMETRIC_SITES),
        "gate_threshold": float(threshold),
        "pressure_method": "orthogonal_l1",
        "pressure_sites": list(EXPECTED_PRESSURE_SITES),
        "pressure_weight": 1.0,
        "step_budget": 1.0,
        "label": f"kappa={float(threshold):g}, lambda=1",
        "is_control": False,
    }


def worker_conditions(config: Mapping[str, Any], worker_id: str) -> list[dict[str, Any]]:
    assignments = mapping(mapping(config, "runpod"), "worker_assignments")
    if worker_id not in assignments:
        raise KeyError(f"Unknown worker {worker_id!r}.")
    by_id = {row["id"]: row for row in condition_specs(config)}
    return [deepcopy(by_id[str(condition_id)]) for condition_id in assignments[worker_id]]


def resolved_condition_config(
    config: Mapping[str, Any], condition: Mapping[str, Any]
) -> dict[str, Any]:
    resolved = deepcopy(dict(config))
    resolved["condition"] = deepcopy(dict(condition))
    resolved["model"]["topology_id"] = "HZ"
    resolved["model"]["site_gate"] = None
    resolved["model"]["site_gates"] = site_gates(float(condition["gate_threshold"]))
    topology, gates = resolve_topology_and_gates(
        "HZ", None, resolved["model"]["site_gates"]
    )
    if topology.active_sites != EXPECTED_ACTIVE_SITES or gates != resolved["model"]["site_gates"]:
        raise RuntimeError("Resolved HZ topology or gates changed.")
    pressure = {
        "method": "orthogonal_l1",
        "sites": list(EXPECTED_PRESSURE_SITES),
        "weight": 1.0,
        "step_budget": 1.0,
        "eps": 1e-12,
    }
    parsed = parse_pressure_config(pressure)
    if not parsed.orthogonal or tuple(parsed.sites) != EXPECTED_PRESSURE_SITES:
        raise RuntimeError("Resolved OL1 pressure changed.")
    resolved["activation_pressure"] = pressure
    return resolved


def run_code_identity() -> dict[str, Any]:
    names = (
        "00_setup_remote.sh", "run041_topology.py", "01_smoke.py", "02_train.py", "03_verify.py",
        "04_monitor.py", "05_remote_preflight.py", "06_build_cache_from_hf.py",
        "_reuse_run004.py", "run041_capture.py", "run_config.py", "initialization.py",
        "optimizer_boundary.py", "diagnostics.py", "smoke.py", "training.py",
        "verification.py",
        "../004-2026-08-29-pythia14m-full-pass-l1n/run_config.py",
        "../004-2026-08-29-pythia14m-full-pass-l1n/initialization.py",
        "../004-2026-08-29-pythia14m-full-pass-l1n/optimizer_boundary.py",
        "../004-2026-08-29-pythia14m-full-pass-l1n/diagnostics.py",
        "../004-2026-08-29-pythia14m-full-pass-l1n/smoke.py",
        "../004-2026-08-29-pythia14m-full-pass-l1n/training.py",
        "../004-2026-08-29-pythia14m-full-pass-l1n/verification.py",
        "../004-2026-08-29-pythia14m-full-pass-l1n/06_build_cache_from_hf.py",
        "../013-2026-08-30-pythia14m-full-pass-a7/artifacts/verification.json",
        "../../src/sparsity_research/artifacts.py",
        "../../src/sparsity_research/capture.py",
        "../../src/sparsity_research/ceilings.py",
        "../../src/sparsity_research/data.py",
        "../../src/sparsity_research/evaluation.py",
        "../../src/sparsity_research/logical_capture.py",
        "../../src/sparsity_research/metrics.py",
        "../../src/sparsity_research/optimization.py",
        "../../src/sparsity_research/pressure.py",
        "../../src/sparsity_research/pythia.py",
        "../../src/sparsity_research/sites.py",
    )
    files = []
    for name in names:
        path = RUN_DIR / name
        # All inventory entries are text. Preserve one source identity across
        # Windows checkout CRLF and the identical Linux Git blob LF contents.
        payload = path.read_bytes().replace(b"\r\n", b"\n")
        files.append({
            "path": name,
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        })
    digest = hashlib.sha256()
    for row in files:
        digest.update(row["path"].encode("utf-8"))
        digest.update(b"\0")
        digest.update(row["sha256"].encode("ascii"))
        digest.update(b"\n")
    return {"files": files, "content_sha256": digest.hexdigest(),
            "text_normalization": "CRLF to LF before byte counts and SHA-256"}


def approved_identity() -> dict[str, Any]:
    config = load_config()
    return {
        "config_sha256": config_sha256(config),
        "run_code": run_code_identity(),
        "expected_initial_parameter_sha256": EXPECTED_INITIAL_PARAMETER_SHA256,
        "expected_training_schedule_sha256": EXPECTED_SCHEDULE_SHA256,
    }


mapping = _BASE.mapping
repo_path = _BASE.repo_path
load_verified_caches = _BASE.load_verified_caches
build_schedule = _BASE.build_schedule
microbatches_for_step = _BASE.microbatches_for_step
require_cuda = _BASE.require_cuda
seed_everything = _BASE.seed_everything
require_training_cache = _BASE.require_training_cache
require_validation_coverage = _BASE.require_validation_coverage
parameter_sha256 = _BASE.parameter_sha256
inventory_content_sha256 = _BASE.inventory_content_sha256
cache_identity = _BASE.cache_identity
git_identity = _BASE.git_identity
write_json = _BASE.write_json

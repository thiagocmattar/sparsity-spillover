"""Approved Run 046 HZ plus h-only OL1 full-pass identities."""

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
from run046_topology import register
register()


RUN_DIR = Path(__file__).resolve().parent
REPO_ROOT = RUN_DIR.parents[1]
DEFAULT_CONFIG = RUN_DIR / "config.yaml"
RUN034_DIR = RUN_DIR.parent / "034-2026-09-17-pythia70m-h-only-ol1"
RUN034_VERIFICATION = RUN034_DIR / "artifacts" / "verification.json"
EXPECTED_THRESHOLDS = (0.5,)
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
EXPECTED_CONDITION_IDS = ("hz-h-ol1-kappa-0p5",)
EXPECTED_RUN034_CONDITION_IDS = ('a4-h-ol1-kappa-0', 'a4-h-ol1-kappa-0p01', 'a4-h-ol1-kappa-0p05', 'a4-h-ol1-kappa-0p1', 'a4-h-ol1-kappa-0p5', 'a7-h-ol1-kappa-0', 'a7-h-ol1-kappa-0p01', 'a7-h-ol1-kappa-0p05', 'a7-h-ol1-kappa-0p1', 'a7-h-ol1-kappa-0p5')
EXPECTED_WORKERS = {condition_id: (condition_id,) for condition_id in EXPECTED_CONDITION_IDS}
EXPECTED_GPU_TYPES = {
    condition_id: "NVIDIA H200" for condition_id in EXPECTED_CONDITION_IDS
}
EXPECTED_INITIAL_PARAMETER_SHA256 = (
    "e8b8d8e48880f8ff25e421ed29b04a81eb417300f2b4a01a8c4d56f2591a1062"
)
EXPECTED_SCHEDULE_SHA256 = (
    "d17a6c0c0d4aacff4b477e6d576f511c12c04ebbc37468f08e6fe61ff1c6ad8e"
)
EXPECTED_RUN034_CODE_SHA256 = (
    "f13879dd7bd521f1547b84ead4239fcb68db0ccc8ef135ef196396974fe11a2d"
)
EXPECTED_CEILING_NUMERATOR = 16_106_127_360
EXPECTED_CEILING_DENOMINATOR = 104_293_466_112
EXPECTED_CEILING_FRACTION = EXPECTED_CEILING_NUMERATOR / EXPECTED_CEILING_DENOMINATOR
EXPECTED_MODEL_CHECKPOINTS = (0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 712)
EXPECTED_OPTIMIZER_CHECKPOINTS = (256, 512, 712)


_BASE = load_run004_module("_run046_frozen_run004_config", "run_config.py")
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
    resolved["model"]["pressure_sites"] = ["h"]
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
        '00_setup_remote.sh',
        'run046_topology.py',
        '02_train.py',
        '03_verify.py',
        '04_monitor.py',
        '05_remote_preflight.py',
        '06_build_cache_from_hf.py',
        '_reuse_run004.py',
        '_reuse_run017.py',
        '_reuse_run018.py',
        'run046_capture.py',
        'run_config.py',
        'initialization.py',
        'initialization_artifact.py',
        'architecture_config.json',
        'model_factory.py',
        'optimizer_boundary.py',
        'diagnostics.py',
        'training.py',
        'verification.py',
        'config.yaml',
        'prelaunch/initialization/metadata.json',
        '../004-2026-08-29-pythia14m-full-pass-l1n/01_smoke.py',
        '../004-2026-08-29-pythia14m-full-pass-l1n/02_train.py',
        '../004-2026-08-29-pythia14m-full-pass-l1n/03_verify.py',
        '../004-2026-08-29-pythia14m-full-pass-l1n/04_monitor.py',
        '../004-2026-08-29-pythia14m-full-pass-l1n/05_remote_preflight.py',
        '../004-2026-08-29-pythia14m-full-pass-l1n/06_build_cache_from_hf.py',
        '../004-2026-08-29-pythia14m-full-pass-l1n/07_plot_spillover_figures.py',
        '../004-2026-08-29-pythia14m-full-pass-l1n/diagnostics.py',
        '../004-2026-08-29-pythia14m-full-pass-l1n/initialization.py',
        '../004-2026-08-29-pythia14m-full-pass-l1n/optimizer_boundary.py',
        '../004-2026-08-29-pythia14m-full-pass-l1n/run_config.py',
        '../004-2026-08-29-pythia14m-full-pass-l1n/smoke.py',
        '../004-2026-08-29-pythia14m-full-pass-l1n/training.py',
        '../004-2026-08-29-pythia14m-full-pass-l1n/verification.py',
        '../017-2026-09-01-pythia70m-selected-ladder-portable-init/01_smoke.py',
        '../017-2026-09-01-pythia70m-selected-ladder-portable-init/02_train.py',
        '../017-2026-09-01-pythia70m-selected-ladder-portable-init/03_verify.py',
        '../017-2026-09-01-pythia70m-selected-ladder-portable-init/04_monitor.py',
        '../017-2026-09-01-pythia70m-selected-ladder-portable-init/05_remote_preflight.py',
        '../017-2026-09-01-pythia70m-selected-ladder-portable-init/06_teal_posthoc.py',
        '../017-2026-09-01-pythia70m-selected-ladder-portable-init/_reuse_run004.py',
        '../017-2026-09-01-pythia70m-selected-ladder-portable-init/diagnostics.py',
        '../017-2026-09-01-pythia70m-selected-ladder-portable-init/initialization.py',
        '../017-2026-09-01-pythia70m-selected-ladder-portable-init/model_factory.py',
        '../017-2026-09-01-pythia70m-selected-ladder-portable-init/optimizer_boundary.py',
        '../017-2026-09-01-pythia70m-selected-ladder-portable-init/run017_capture.py',
        '../017-2026-09-01-pythia70m-selected-ladder-portable-init/run_config.py',
        '../017-2026-09-01-pythia70m-selected-ladder-portable-init/smoke.py',
        '../017-2026-09-01-pythia70m-selected-ladder-portable-init/teal_posthoc.py',
        '../017-2026-09-01-pythia70m-selected-ladder-portable-init/training.py',
        '../017-2026-09-01-pythia70m-selected-ladder-portable-init/verification.py',
        '../018-2026-09-01-pythia70m-selected-ladder-canonical-init/00_generate_initialization.py',
        '../018-2026-09-01-pythia70m-selected-ladder-canonical-init/02_smoke.py',
        '../018-2026-09-01-pythia70m-selected-ladder-canonical-init/03_train.py',
        '../018-2026-09-01-pythia70m-selected-ladder-canonical-init/04_verify.py',
        '../018-2026-09-01-pythia70m-selected-ladder-canonical-init/05_monitor.py',
        '../018-2026-09-01-pythia70m-selected-ladder-canonical-init/06_remote_preflight.py',
        '../018-2026-09-01-pythia70m-selected-ladder-canonical-init/07_teal_posthoc.py',
        '../018-2026-09-01-pythia70m-selected-ladder-canonical-init/08_verify_observed_bound.py',
        '../018-2026-09-01-pythia70m-selected-ladder-canonical-init/09_teal_posthoc_observed_bound.py',
        '../018-2026-09-01-pythia70m-selected-ladder-canonical-init/_reuse_run004.py',
        '../018-2026-09-01-pythia70m-selected-ladder-canonical-init/_reuse_run017.py',
        '../018-2026-09-01-pythia70m-selected-ladder-canonical-init/diagnostics.py',
        '../018-2026-09-01-pythia70m-selected-ladder-canonical-init/initialization.py',
        '../018-2026-09-01-pythia70m-selected-ladder-canonical-init/initialization_artifact.py',
        '../018-2026-09-01-pythia70m-selected-ladder-canonical-init/model_factory.py',
        '../018-2026-09-01-pythia70m-selected-ladder-canonical-init/optimizer_boundary.py',
        '../018-2026-09-01-pythia70m-selected-ladder-canonical-init/run018_capture.py',
        '../018-2026-09-01-pythia70m-selected-ladder-canonical-init/run_config.py',
        '../018-2026-09-01-pythia70m-selected-ladder-canonical-init/smoke.py',
        '../018-2026-09-01-pythia70m-selected-ladder-canonical-init/teal_posthoc.py',
        '../018-2026-09-01-pythia70m-selected-ladder-canonical-init/training.py',
        '../018-2026-09-01-pythia70m-selected-ladder-canonical-init/verification.py',
        '../018-2026-09-01-pythia70m-selected-ladder-canonical-init/verification_observed_bound.py',
        '../034-2026-09-17-pythia70m-h-only-ol1/artifacts/verification.json',
        '../../src/sparsity_research/__init__.py',
        '../../src/sparsity_research/artifacts.py',
        '../../src/sparsity_research/capture.py',
        '../../src/sparsity_research/ceilings.py',
        '../../src/sparsity_research/clipping.py',
        '../../src/sparsity_research/data.py',
        '../../src/sparsity_research/evaluation.py',
        '../../src/sparsity_research/logical_capture.py',
        '../../src/sparsity_research/metrics.py',
        '../../src/sparsity_research/optimization.py',
        '../../src/sparsity_research/pressure.py',
        '../../src/sparsity_research/pythia.py',
        '../../src/sparsity_research/sites.py',
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

validate_science_config = validate_config

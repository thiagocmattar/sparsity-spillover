"""Approved Run 012 A4-Z plus OL1 conditions and frozen full-pass identities."""

from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path
from typing import Any, Mapping

import yaml

from sparsity_research.artifacts import config_sha256
from sparsity_research.pressure import parse_pressure_config
from sparsity_research.sites import resolve_topology_and_gate

from _reuse_run004 import load_run004_module


RUN_DIR = Path(__file__).resolve().parent
REPO_ROOT = RUN_DIR.parents[1]
DEFAULT_CONFIG = RUN_DIR / "config.yaml"
RUN011_DIR = RUN_DIR.parent / "011-2026-08-30-pythia14m-full-pass-a4z"
RUN011_VERIFICATION = RUN011_DIR / "artifacts" / "verification.json"
EXPECTED_THRESHOLDS = (0.0, 0.01, 0.05, 0.1, 0.5)
EXPECTED_ACTIVE_SITES = ("a", "m", "h", "z")
EXPECTED_CONDITION_IDS = (
    "a4z-ol1-kappa-0",
    "a4z-ol1-kappa-0p01",
    "a4z-ol1-kappa-0p05",
    "a4z-ol1-kappa-0p1",
    "a4z-ol1-kappa-0p5",
)
EXPECTED_RUN011_CONDITION_IDS = (
    "a4z-one-sided-kappa-0",
    "a4z-one-sided-kappa-0p01",
    "a4z-one-sided-kappa-0p05",
    "a4z-one-sided-kappa-0p1",
    "a4z-one-sided-kappa-0p5",
)
EXPECTED_WORKERS = {condition_id: (condition_id,) for condition_id in EXPECTED_CONDITION_IDS}
EXPECTED_GPU_TYPES = {
    condition_id: "NVIDIA A100-SXM4-80GB" for condition_id in EXPECTED_CONDITION_IDS
}
EXPECTED_INITIAL_PARAMETER_SHA256 = (
    "ece58512e94ee2f97d17278fe8af4c1abef9c5f7f9dbdd4087e36d7f67d7af57"
)
EXPECTED_SCHEDULE_SHA256 = (
    "f1755812b4f70806bd137ee900c9338f64c4c2074b6dd8b7661e6bde9b141faa"
)
EXPECTED_RUN011_CODE_SHA256 = (
    "c253d6cba5511e7e43f3599a5776b846b4a8b8048a32c23fcfc66317d8b1bf09"
)
EXPECTED_CEILING_NUMERATOR = 2_415_919_104
EXPECTED_CEILING_DENOMINATOR = 18_825_609_216
EXPECTED_CEILING_FRACTION = EXPECTED_CEILING_NUMERATOR / EXPECTED_CEILING_DENOMINATOR
EXPECTED_MODEL_CHECKPOINTS = (0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 712)
EXPECTED_OPTIMIZER_CHECKPOINTS = (256, 512, 712)


_BASE = load_run004_module("_run012_frozen_run004_config", "run_config.py")
_BASE.RUN_DIR = RUN_DIR
_BASE.REPO_ROOT = REPO_ROOT
_BASE.DEFAULT_CONFIG = DEFAULT_CONFIG


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Run config must be a mapping.")
    validate_config(config)
    return config


def validate_config(config: Mapping[str, Any]) -> None:
    """Validate the Run 004 recipe, then the approved A4-Z plus OL1 delta."""

    proxy = deepcopy(dict(config))
    proxy["conditions"] = {
        "pressure_method": "l1_naive",
        "pressure_site": "h",
        "relu_pressure_weights": [0.05, 0.1, 0.5, 1.0],
        "include_gelu_control": True,
        "include_relu_control": True,
    }
    proxy_diagnostics = deepcopy(dict(mapping(config, "diagnostics")))
    proxy_diagnostics["activation_sites"] = [
        "m", "h", "q_post", "k_post", "v", "attention_output"
    ]
    proxy["diagnostics"] = proxy_diagnostics
    proxy_runpod = deepcopy(dict(mapping(config, "runpod")))
    proxy_runpod["pod_count"] = 5
    proxy_runpod["terminate_after_hours"] = 7
    proxy_runpod["worker_assignments"] = {
        "controls": ["gelu-control", "relu-control"],
        "relu-l1n-0p05": ["relu-l1n-0p05"],
        "relu-l1n-0p1": ["relu-l1n-0p1"],
        "relu-l1n-0p5": ["relu-l1n-0p5"],
        "relu-l1n-1": ["relu-l1n-1"],
    }
    proxy["runpod"] = proxy_runpod
    _BASE.validate_config(proxy)

    model = mapping(config, "model")
    if (
        model.get("architecture") != "EleutherAI/pythia-14m-deduped"
        or model.get("revision") != "7386d9a4ae45aef494a6e704910394def3037fc5"
        or model.get("initialization") != "random"
    ):
        raise ValueError("Run 012 requires the pinned random Pythia-14M architecture config.")
    expected_gate = {"operator": "one_sided_threshold", "kappa": 0.0}
    if model.get("topology_id") != "A4-Z" or model.get("site_gate") != expected_gate:
        raise ValueError("The base Run 012 model must declare A4-Z at kappa=0.")
    topology, gate = resolve_topology_and_gate(str(model["topology_id"]), dict(model["site_gate"]))
    if topology.active_sites != EXPECTED_ACTIVE_SITES or gate != expected_gate:
        raise ValueError("A4-Z did not resolve to the approved sites and gate.")

    conditions = mapping(config, "conditions")
    if tuple(conditions.get("active_sites", ())) != EXPECTED_ACTIVE_SITES:
        raise ValueError("The approved joint A4-Z sites changed.")
    if conditions.get("gate_operator") != "one_sided_threshold":
        raise ValueError("Run 012 requires the one-sided threshold gate.")
    if tuple(float(value) for value in conditions.get("gate_thresholds", ())) != EXPECTED_THRESHOLDS:
        raise ValueError("The approved paper-scale kappa grid changed.")
    if conditions.get("pressure_method") != "orthogonal_l1":
        raise ValueError("Run 012 requires operational orthogonal_l1.")
    if tuple(conditions.get("pressure_sites", ())) != EXPECTED_ACTIVE_SITES:
        raise ValueError("OL1 pressure must target every gated A4-Z output.")
    if float(conditions.get("pressure_weight", -1.0)) != 1.0:
        raise ValueError("Run 012 fixes OL1 lambda at 1.0.")
    if float(conditions.get("step_budget", -1.0)) != 1.0:
        raise ValueError("Run 012 fixes the OL1 trust budget at 1.0.")
    if tuple(row["id"] for row in condition_specs(config)) != EXPECTED_CONDITION_IDS:
        raise ValueError("The five-condition Run 012 identity/order changed.")

    training = mapping(config, "training")
    if training.get("fp16_ol1_overflow_policy") != "skip_entire_boundary":
        raise ValueError("FP16 overflow must skip both AdamW and the OL1 correction.")

    diagnostics = mapping(config, "diagnostics")
    expected_sites = ("a", "m", "h", "q_post", "k_post", "v", "z", "attention_output")
    if tuple(diagnostics.get("activation_sites", ())) != expected_sites:
        raise ValueError("The approved terminal activation sites changed.")
    if tuple(float(value) for value in diagnostics.get("near_zero_thresholds", ())) != (
        0.0, 0.001, 0.01
    ):
        raise ValueError("The approved near-zero thresholds changed.")
    if diagnostics.get("gradient_interaction") is not True:
        raise ValueError("Training-time gradient interaction is required for OL1.")
    if diagnostics.get("ol1_adaptive_directions") is not True:
        raise ValueError("OL1 adaptive-direction metrics are required.")
    if diagnostics.get("clipping_frontier") is not None:
        raise ValueError("A clipping frontier was not approved.")

    checkpoints = mapping(config, "checkpoints")
    if tuple(int(value) for value in checkpoints.get("model_steps", ())) != EXPECTED_MODEL_CHECKPOINTS:
        raise ValueError("Run 004 model-checkpoint cadence changed.")
    if tuple(int(value) for value in checkpoints.get("optimizer_steps", ())) != EXPECTED_OPTIMIZER_CHECKPOINTS:
        raise ValueError("Run 004 recovery-checkpoint cadence changed.")

    comparison = mapping(config, "comparison")
    expected_comparison = {
        "source_run": "runs/011-2026-08-30-pythia14m-full-pass-a4z",
        "verification": (
            "runs/011-2026-08-30-pythia14m-full-pass-a4z/artifacts/verification.json"
        ),
        "matched_condition_ids": list(EXPECTED_RUN011_CONDITION_IDS),
        "initial_parameter_sha256": EXPECTED_INITIAL_PARAMETER_SHA256,
        "training_schedule_sha256": EXPECTED_SCHEDULE_SHA256,
        "run011_code_sha256": EXPECTED_RUN011_CODE_SHA256,
    }
    if dict(comparison) != expected_comparison:
        raise ValueError("Run 011 comparator identity changed.")

    runpod = mapping(config, "runpod")
    assignments = {
        str(key): tuple(str(item) for item in value)
        for key, value in mapping(runpod, "worker_assignments").items()
    }
    gpu_types = {
        str(key): str(value)
        for key, value in mapping(runpod, "gpu_type_by_worker").items()
    }
    if int(runpod.get("pod_count", 0)) != 5 or assignments != EXPECTED_WORKERS:
        raise ValueError("Run 012 requires five independent one-condition Pods.")
    if runpod.get("parallelism") != "one_condition_per_gpu":
        raise ValueError("Run 012 parallelism must remain condition-level, not DDP.")
    if gpu_types != EXPECTED_GPU_TYPES:
        raise ValueError("The approved A100 SXM4 worker targets changed.")
    if runpod.get("cloud_type") != "SECURE":
        raise ValueError("The approved 80 GB A100 target requires Secure Cloud.")
    if float(runpod.get("seed_worker_terminate_after_hours", 0.0)) != 4.0:
        raise ValueError("The seed worker must have a four-hour absolute guard.")
    if float(runpod.get("additional_worker_terminate_after_hours", 0.0)) != 2.5:
        raise ValueError("Every additional worker must have a 2.5-hour absolute guard.")

    artifacts = mapping(config, "artifacts")
    if (
        artifacts.get("save_final_checkpoint") is not True
        or artifacts.get("save_optimizer_at_declared_steps") is not True
        or artifacts.get("retain_predictions") is not False
    ):
        raise ValueError("Run 012 checkpoint and prediction retention changed.")


def condition_specs(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    thresholds = [float(value) for value in mapping(config, "conditions")["gate_thresholds"]]
    return [_condition(order, threshold) for order, threshold in enumerate(thresholds, 1)]


def _condition(order: int, threshold: float) -> dict[str, Any]:
    token = f"{float(threshold):g}".replace(".", "p")
    return {
        "id": f"a4z-ol1-kappa-{token}",
        "order": int(order),
        "topology_id": "A4-Z",
        "active_sites": list(EXPECTED_ACTIVE_SITES),
        "gate_operator": "one_sided_threshold",
        "gate_threshold": float(threshold),
        "pressure_method": "orthogonal_l1",
        "pressure_sites": list(EXPECTED_ACTIVE_SITES),
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
    topology_id = "A4-Z"
    site_gate = {
        "operator": "one_sided_threshold",
        "kappa": float(condition["gate_threshold"]),
    }
    topology, gate = resolve_topology_and_gate(topology_id, site_gate)
    if topology.active_sites != EXPECTED_ACTIVE_SITES or gate != site_gate:
        raise RuntimeError("Resolved A4-Z topology or gate changed.")
    resolved["model"]["topology_id"] = topology_id
    resolved["model"]["site_gate"] = site_gate
    pressure = {
        "method": "orthogonal_l1",
        "sites": list(EXPECTED_ACTIVE_SITES),
        "weight": 1.0,
        "step_budget": 1.0,
        "eps": 1e-12,
    }
    parsed = parse_pressure_config(pressure)
    if not parsed.orthogonal or tuple(parsed.sites) != EXPECTED_ACTIVE_SITES:
        raise RuntimeError("Resolved OL1 pressure changed.")
    resolved["activation_pressure"] = pressure
    return resolved


def run_code_identity() -> dict[str, Any]:
    names = (
        "00_setup_remote.sh", "01_smoke.py", "02_train.py", "03_verify.py",
        "04_monitor.py", "05_remote_preflight.py", "06_build_cache_from_hf.py",
        "_reuse_run004.py", "run_config.py", "initialization.py",
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
        payload = path.read_bytes()
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
    return {"files": files, "content_sha256": digest.hexdigest()}


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

"""Approved Run 013 A7-Z-POST conditions and frozen full-pass identities."""

from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path
from typing import Any, Mapping

import yaml

from sparsity_research.artifacts import config_sha256
from sparsity_research.pressure import parse_pressure_config
from sparsity_research.sites import resolve_topology_and_gates

from _reuse_run004 import RUN004_DIR, load_run004_module
from _reuse_run011 import RUN011_DIR, load_run011_module


RUN_DIR = Path(__file__).resolve().parent
REPO_ROOT = RUN_DIR.parents[1]
DEFAULT_CONFIG = RUN_DIR / "config.yaml"
RUN011_VERIFICATION = RUN011_DIR / "artifacts" / "verification.json"
EXPECTED_THRESHOLDS = (0.0, 0.01, 0.05, 0.1, 0.5)
EXPECTED_ACTIVE_SITES = ("a", "m", "h", "q_post", "k_post", "v", "z")
EXPECTED_ONE_SIDED_SITES = ("a", "m", "h", "z")
EXPECTED_SYMMETRIC_SITES = ("q_post", "k_post", "v")
EXPECTED_CONDITION_IDS = (
    "a7-z-post-mixed-kappa-0",
    "a7-z-post-mixed-kappa-0p01",
    "a7-z-post-mixed-kappa-0p05",
    "a7-z-post-mixed-kappa-0p1",
    "a7-z-post-mixed-kappa-0p5",
)
EXPECTED_A4_CONDITION_IDS = (
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
EXPECTED_RUN004_CODE_SHA256 = (
    "501744d66aec47c469c04a1885c97372b39315bd3bc2a67297d8353e4efe5e2d"
)
EXPECTED_RUN011_CODE_SHA256 = (
    "c253d6cba5511e7e43f3599a5776b846b4a8b8048a32c23fcfc66317d8b1bf09"
)
EXPECTED_CEILING_NUMERATOR = 5_638_717_440
EXPECTED_CEILING_DENOMINATOR = 18_825_609_216
EXPECTED_CEILING_FRACTION = EXPECTED_CEILING_NUMERATOR / EXPECTED_CEILING_DENOMINATOR
EXPECTED_MODEL_CHECKPOINTS = (0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 712)
EXPECTED_OPTIMIZER_CHECKPOINTS = (256, 512, 712)


_BASE = load_run011_module("_run013_frozen_run011_config", "run_config.py")
_BASE.RUN_DIR = RUN_DIR
_BASE.REPO_ROOT = REPO_ROOT
_BASE.DEFAULT_CONFIG = DEFAULT_CONFIG


def site_gates(kappa: float) -> dict[str, dict[str, Any]]:
    value = float(kappa)
    one_sided = {"operator": "one_sided_threshold", "kappa": value}
    symmetric = {"operator": "symmetric_threshold", "kappa": value}
    return {
        site: dict(one_sided if site in EXPECTED_ONE_SIDED_SITES else symmetric)
        for site in EXPECTED_ACTIVE_SITES
    }


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Run config must be a mapping.")
    validate_config(config)
    return config


def validate_config(config: Mapping[str, Any]) -> None:
    """Validate the frozen Run 011 full-pass recipe, then the approved A7 delta."""

    proxy = deepcopy(dict(config))
    proxy_model = deepcopy(dict(mapping(config, "model")))
    proxy_model["topology_id"] = "A4-Z"
    proxy_model["site_gate"] = {"operator": "one_sided_threshold", "kappa": 0.0}
    proxy_model.pop("site_gates", None)
    proxy["model"] = proxy_model
    proxy["conditions"] = {
        "active_sites": ["a", "m", "h", "z"],
        "gate_operator": "one_sided_threshold",
        "gate_thresholds": list(EXPECTED_THRESHOLDS),
        "pressure_method": "none",
    }
    proxy["comparison"] = {
        "source_run": "runs/004-2026-08-29-pythia14m-full-pass-l1n",
        "verification": (
            "runs/004-2026-08-29-pythia14m-full-pass-l1n/artifacts/verification.json"
        ),
        "reused_conditions": ["relu-control"],
        "initial_parameter_sha256": EXPECTED_INITIAL_PARAMETER_SHA256,
        "training_schedule_sha256": EXPECTED_SCHEDULE_SHA256,
        "run004_code_sha256": EXPECTED_RUN004_CODE_SHA256,
    }
    proxy_runpod = deepcopy(dict(mapping(config, "runpod")))
    proxy_runpod["worker_assignments"] = {
        condition_id: [condition_id] for condition_id in EXPECTED_A4_CONDITION_IDS
    }
    proxy_runpod["gpu_type_by_worker"] = {
        condition_id: "NVIDIA A100-SXM4-80GB"
        for condition_id in EXPECTED_A4_CONDITION_IDS
    }
    proxy["runpod"] = proxy_runpod
    _BASE.validate_config(proxy)

    model = mapping(config, "model")
    if (
        model.get("architecture") != "EleutherAI/pythia-14m-deduped"
        or model.get("revision") != "7386d9a4ae45aef494a6e704910394def3037fc5"
        or model.get("initialization") != "random"
    ):
        raise ValueError("Run 013 requires the pinned random Pythia-14M architecture config.")
    expected_gates = site_gates(0.0)
    if (
        model.get("topology_id") != "A7-Z-POST"
        or model.get("site_gate") is not None
        or model.get("site_gates") != expected_gates
    ):
        raise ValueError("The base Run 013 model must declare mixed A7 gates at kappa=0.")
    topology, gates = resolve_topology_and_gates(
        str(model["topology_id"]), model.get("site_gate"), model.get("site_gates")
    )
    if topology.active_sites != EXPECTED_ACTIVE_SITES or gates != expected_gates:
        raise ValueError("A7-Z-POST did not resolve to the approved sites and gates.")

    conditions = mapping(config, "conditions")
    if tuple(conditions.get("active_sites", ())) != EXPECTED_ACTIVE_SITES:
        raise ValueError("The approved seven active sites changed.")
    if tuple(conditions.get("one_sided_sites", ())) != EXPECTED_ONE_SIDED_SITES:
        raise ValueError("The approved one-sided sites changed.")
    if tuple(conditions.get("symmetric_sites", ())) != EXPECTED_SYMMETRIC_SITES:
        raise ValueError("The approved symmetric sites changed.")
    if tuple(float(value) for value in conditions.get("gate_thresholds", ())) != EXPECTED_THRESHOLDS:
        raise ValueError("The approved paper-scale kappa grid changed.")
    if conditions.get("pressure_method") != "none":
        raise ValueError("Run 013 must not use L1 or OL1 pressure.")
    if tuple(row["id"] for row in condition_specs(config)) != EXPECTED_CONDITION_IDS:
        raise ValueError("The five-condition Run 013 identity/order changed.")

    comparison = mapping(config, "comparison")
    expected_comparison = {
        "source_run": "runs/011-2026-08-30-pythia14m-full-pass-a4z",
        "verification": (
            "runs/011-2026-08-30-pythia14m-full-pass-a4z/artifacts/verification.json"
        ),
        "matched_condition_ids": list(EXPECTED_A4_CONDITION_IDS),
        "initial_parameter_sha256": EXPECTED_INITIAL_PARAMETER_SHA256,
        "training_schedule_sha256": EXPECTED_SCHEDULE_SHA256,
        "run011_code_sha256": EXPECTED_RUN011_CODE_SHA256,
    }
    if dict(comparison) != expected_comparison:
        raise ValueError("Run 011 matched A4 comparator identity changed.")

    diagnostics = mapping(config, "diagnostics")
    expected_sites = ("a", "m", "h", "q_post", "k_post", "v", "z", "attention_output")
    if tuple(diagnostics.get("activation_sites", ())) != expected_sites:
        raise ValueError("The approved terminal activation sites changed.")
    if tuple(float(value) for value in diagnostics.get("near_zero_thresholds", ())) != (
        0.0, 0.001, 0.01
    ):
        raise ValueError("The approved near-zero thresholds changed.")
    if diagnostics.get("gradient_interaction") is not False:
        raise ValueError("Gradient-interaction metrics are inapplicable without pressure.")
    if diagnostics.get("clipping_frontier") is not None:
        raise ValueError("A clipping frontier was not approved.")

    checkpoints = mapping(config, "checkpoints")
    if tuple(int(value) for value in checkpoints.get("model_steps", ())) != EXPECTED_MODEL_CHECKPOINTS:
        raise ValueError("The paper-scale model-checkpoint cadence changed.")
    if tuple(int(value) for value in checkpoints.get("optimizer_steps", ())) != EXPECTED_OPTIMIZER_CHECKPOINTS:
        raise ValueError("The paper-scale recovery-checkpoint cadence changed.")

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
        raise ValueError("Run 013 requires five independent one-condition Pods.")
    if runpod.get("parallelism") != "one_condition_per_gpu":
        raise ValueError("Run 013 parallelism must remain condition-level, not DDP.")
    if gpu_types != EXPECTED_GPU_TYPES:
        raise ValueError("The prelaunch A100 SXM worker targets changed.")
    if runpod.get("cloud_type") != "SECURE":
        raise ValueError("The approved 80 GB A100 target requires Secure Cloud.")
    if float(runpod.get("preflight_terminate_after_hours", 0.0)) != 1.5:
        raise ValueError("The preflight Pod must have a 1.5-hour termination guard.")
    if float(runpod.get("scientific_terminate_after_hours", 0.0)) != 2.5:
        raise ValueError("Every scientific Pod must have a 2.5-hour termination guard.")

    artifacts = mapping(config, "artifacts")
    if (
        artifacts.get("save_final_checkpoint") is not True
        or artifacts.get("save_optimizer_at_declared_steps") is not True
        or artifacts.get("retain_predictions") is not False
    ):
        raise ValueError("Run 013 checkpoint and prediction retention changed.")


def condition_specs(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    thresholds = [float(value) for value in mapping(config, "conditions")["gate_thresholds"]]
    return [_condition(order, threshold) for order, threshold in enumerate(thresholds, 1)]


def _condition(order: int, threshold: float) -> dict[str, Any]:
    token = f"{float(threshold):g}".replace(".", "p")
    return {
        "id": f"a7-z-post-mixed-kappa-{token}",
        "order": int(order),
        "topology_id": "A7-Z-POST",
        "active_sites": list(EXPECTED_ACTIVE_SITES),
        "one_sided_sites": list(EXPECTED_ONE_SIDED_SITES),
        "symmetric_sites": list(EXPECTED_SYMMETRIC_SITES),
        "gate_threshold": float(threshold),
        "pressure_method": "none",
        "pressure_sites": [],
        "pressure_weight": 0.0,
        "step_budget": None,
        "label": f"kappa={float(threshold):g}",
        "is_control": float(threshold) == 0.0,
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
    resolved["model"]["topology_id"] = "A7-Z-POST"
    resolved["model"]["site_gate"] = None
    resolved["model"]["site_gates"] = site_gates(float(condition["gate_threshold"]))
    topology, gates = resolve_topology_and_gates(
        "A7-Z-POST", None, resolved["model"]["site_gates"]
    )
    if topology.active_sites != EXPECTED_ACTIVE_SITES or gates != resolved["model"]["site_gates"]:
        raise RuntimeError("Resolved A7-Z-POST topology or gates changed.")
    pressure = {
        "method": "none",
        "sites": [],
        "weight": 0.0,
        "step_budget": None,
        "eps": 1e-12,
    }
    parse_pressure_config(pressure)
    resolved["activation_pressure"] = pressure
    return resolved


def run_code_identity() -> dict[str, Any]:
    names = (
        "00_setup_remote.sh", "01_smoke.py", "02_train.py", "03_verify.py",
        "04_monitor.py", "05_remote_preflight.py", "06_build_cache_from_hf.py",
        "_reuse_run004.py", "_reuse_run011.py", "run_config.py", "initialization.py",
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
        "../011-2026-08-30-pythia14m-full-pass-a4z/run_config.py",
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

"""Approved Run 009 OL1 conditions, Run 004 matches, and immutable identities."""

from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path
from typing import Any, Mapping

import yaml

from sparsity_research.artifacts import config_sha256
from sparsity_research.pressure import parse_pressure_config
from sparsity_research.sites import resolve_topology_and_gate

from _reuse_run004 import RUN004_DIR, load_run004_module


RUN_DIR = Path(__file__).resolve().parent
REPO_ROOT = RUN_DIR.parents[1]
DEFAULT_CONFIG = RUN_DIR / "config.yaml"
RUN004_VERIFICATION = RUN004_DIR / "artifacts" / "verification.json"
EXPECTED_WEIGHTS = (0.05, 0.1, 0.5, 1.0)
EXPECTED_CONDITION_IDS = (
    "relu-ol1-0p05",
    "relu-ol1-0p1",
    "relu-ol1-0p5",
    "relu-ol1-1",
)
EXPECTED_WORKERS = {condition_id: (condition_id,) for condition_id in EXPECTED_CONDITION_IDS}
EXPECTED_GPU_TYPES = {
    "relu-ol1-0p05": "NVIDIA A100-SXM4-80GB",
    "relu-ol1-0p1": "NVIDIA A100-SXM4-80GB",
    "relu-ol1-0p5": "NVIDIA A100-SXM4-80GB",
    "relu-ol1-1": "NVIDIA A100 80GB PCIe",
}
EXPECTED_INITIAL_PARAMETER_SHA256 = "ece58512e94ee2f97d17278fe8af4c1abef9c5f7f9dbdd4087e36d7f67d7af57"
EXPECTED_SCHEDULE_SHA256 = "f1755812b4f70806bd137ee900c9338f64c4c2074b6dd8b7661e6bde9b141faa"
EXPECTED_RUN004_CODE_SHA256 = "501744d66aec47c469c04a1885c97372b39315bd3bc2a67297d8353e4efe5e2d"
EXPECTED_MODEL_CHECKPOINTS = (0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 712)
EXPECTED_OPTIMIZER_CHECKPOINTS = (256, 512, 712)


_BASE = load_run004_module("_run009_frozen_run004_config", "run_config.py")
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
    """Validate the Run 004 match first, then the approved OL1-only delta."""

    proxy = deepcopy(dict(config))
    proxy["conditions"] = {
        "pressure_method": "l1_naive",
        "pressure_site": "h",
        "relu_pressure_weights": list(EXPECTED_WEIGHTS),
        "include_gelu_control": True,
        "include_relu_control": True,
    }
    proxy_runpod = deepcopy(dict(mapping(config, "runpod")))
    proxy_runpod["pod_count"] = 5
    proxy_runpod["worker_assignments"] = {
        "controls": ["gelu-control", "relu-control"],
        "relu-l1n-0p05": ["relu-l1n-0p05"],
        "relu-l1n-0p1": ["relu-l1n-0p1"],
        "relu-l1n-0p5": ["relu-l1n-0p5"],
        "relu-l1n-1": ["relu-l1n-1"],
    }
    proxy["runpod"] = proxy_runpod
    _BASE.validate_config(proxy)

    conditions = mapping(config, "conditions")
    if conditions.get("pressure_method") != "orthogonal_l1":
        raise ValueError("Run 009 requires operational orthogonal_l1.")
    if conditions.get("pressure_site") != "h":
        raise ValueError("Run 009 pressure must remain h-only.")
    if tuple(float(value) for value in conditions.get("relu_pressure_weights", ())) != EXPECTED_WEIGHTS:
        raise ValueError("Run 009 uses the approved Run 004 lambda grid.")
    if float(conditions.get("step_budget", -1.0)) != 1.0:
        raise ValueError("Run 009 OL1 requires the approved trust budget 1.0.")
    if conditions.get("include_controls") is not False:
        raise ValueError("Run 009 must reuse, not rerun, Run 004 controls.")
    if tuple(row["id"] for row in condition_specs(config)) != EXPECTED_CONDITION_IDS:
        raise ValueError("The four-condition Run 009 identity/order changed.")

    training = mapping(config, "training")
    if training.get("fp16_ol1_overflow_policy") != "skip_entire_boundary":
        raise ValueError("FP16 overflow must skip both AdamW and the OL1 correction.")
    checkpoints = mapping(config, "checkpoints")
    if tuple(int(value) for value in checkpoints.get("model_steps", ())) != EXPECTED_MODEL_CHECKPOINTS:
        raise ValueError("Run 004 model-checkpoint cadence changed.")
    if tuple(int(value) for value in checkpoints.get("optimizer_steps", ())) != EXPECTED_OPTIMIZER_CHECKPOINTS:
        raise ValueError("Run 004 recovery-checkpoint cadence changed.")

    comparison = mapping(config, "comparison")
    expected_comparison = {
        "source_run": "runs/004-2026-08-29-pythia14m-full-pass-l1n",
        "verification": "runs/004-2026-08-29-pythia14m-full-pass-l1n/artifacts/verification.json",
        "reused_controls": ["gelu-control", "relu-control"],
        "matched_l1_conditions": [
            "relu-l1n-0p05", "relu-l1n-0p1", "relu-l1n-0p5", "relu-l1n-1"
        ],
        "initial_parameter_sha256": EXPECTED_INITIAL_PARAMETER_SHA256,
        "training_schedule_sha256": EXPECTED_SCHEDULE_SHA256,
        "run004_code_sha256": EXPECTED_RUN004_CODE_SHA256,
    }
    if dict(comparison) != expected_comparison:
        raise ValueError("Run 004 comparator identity changed.")

    runpod = mapping(config, "runpod")
    assignments = {
        str(key): tuple(str(item) for item in value)
        for key, value in mapping(runpod, "worker_assignments").items()
    }
    gpu_types = {str(key): str(value) for key, value in mapping(runpod, "gpu_type_by_worker").items()}
    if int(runpod.get("pod_count", 0)) != 4 or assignments != EXPECTED_WORKERS:
        raise ValueError("Run 009 requires four independent one-condition Pods.")
    if gpu_types != EXPECTED_GPU_TYPES:
        raise ValueError("Pairwise Run 004 GPU matching changed.")
    if runpod.get("cloud_type") != "SECURE":
        raise ValueError("The approved 80 GB A100 plan requires Secure Cloud.")


def condition_specs(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    weights = [float(value) for value in mapping(config, "conditions")["relu_pressure_weights"]]
    return [_condition(order, weight) for order, weight in enumerate(weights, 1)]


def _condition(order: int, weight: float) -> dict[str, Any]:
    token = f"{float(weight):g}".replace(".", "p")
    return {
        "id": f"relu-ol1-{token}",
        "order": int(order),
        "activation": "relu",
        "pressure_method": "orthogonal_l1",
        "pressure_sites": ["h"],
        "pressure_weight": float(weight),
        "step_budget": 1.0,
        "label": f"lambda={float(weight):g}",
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
    topology_id, site_gate = "A1-H", {"operator": "relu"}
    resolve_topology_and_gate(topology_id, site_gate)
    resolved["model"]["topology_id"] = topology_id
    resolved["model"]["site_gate"] = site_gate
    pressure = {
        "method": "orthogonal_l1",
        "sites": ["h"],
        "weight": float(condition["pressure_weight"]),
        "step_budget": 1.0,
        "eps": 1e-12,
    }
    parse_pressure_config(pressure)
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

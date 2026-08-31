"""Corrected Run 015 A4-Z plus verified four-site OL1 identities."""

from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path
from typing import Any, Mapping

import yaml

from sparsity_research.artifacts import config_sha256

from _reuse_run012 import load_run012_module


RUN_DIR = Path(__file__).resolve().parent
REPO_ROOT = RUN_DIR.parents[1]
DEFAULT_CONFIG = RUN_DIR / "config.yaml"
RUN011_DIR = RUN_DIR.parent / "011-2026-08-30-pythia14m-full-pass-a4z"
RUN011_VERIFICATION = RUN011_DIR / "artifacts" / "verification.json"

_BASE = load_run012_module("_run015_valid_run012_config", "run_config.py")
_BASE.RUN_DIR = RUN_DIR
_BASE.REPO_ROOT = REPO_ROOT
_BASE.DEFAULT_CONFIG = DEFAULT_CONFIG
_BASE.RUN011_DIR = RUN011_DIR
_BASE.RUN011_VERIFICATION = RUN011_VERIFICATION

EXPECTED_THRESHOLDS = _BASE.EXPECTED_THRESHOLDS
EXPECTED_ACTIVE_SITES = _BASE.EXPECTED_ACTIVE_SITES
EXPECTED_CONDITION_IDS = _BASE.EXPECTED_CONDITION_IDS
EXPECTED_RUN011_CONDITION_IDS = _BASE.EXPECTED_RUN011_CONDITION_IDS
EXPECTED_WORKERS = _BASE.EXPECTED_WORKERS
EXPECTED_GPU_TYPES = _BASE.EXPECTED_GPU_TYPES
EXPECTED_INITIAL_PARAMETER_SHA256 = _BASE.EXPECTED_INITIAL_PARAMETER_SHA256
EXPECTED_SCHEDULE_SHA256 = _BASE.EXPECTED_SCHEDULE_SHA256
EXPECTED_RUN011_CODE_SHA256 = _BASE.EXPECTED_RUN011_CODE_SHA256
EXPECTED_CEILING_NUMERATOR = _BASE.EXPECTED_CEILING_NUMERATOR
EXPECTED_CEILING_DENOMINATOR = _BASE.EXPECTED_CEILING_DENOMINATOR
EXPECTED_CEILING_FRACTION = _BASE.EXPECTED_CEILING_FRACTION
EXPECTED_MODEL_CHECKPOINTS = _BASE.EXPECTED_MODEL_CHECKPOINTS
EXPECTED_OPTIMIZER_CHECKPOINTS = _BASE.EXPECTED_OPTIMIZER_CHECKPOINTS
EXPECTED_PRESSURE_CAPTURE_TENSOR_COUNT = 24
EXPECTED_PRESSURE_CAPTURE_NAMES = tuple(
    sorted(
        f"{site}.layer_{layer}"
        for site in EXPECTED_ACTIVE_SITES
        for layer in range(6)
    )
)
EXPECTED_PRESSURE_CAPTURE_NAMES_SHA256 = hashlib.sha256(
    "\n".join(EXPECTED_PRESSURE_CAPTURE_NAMES).encode("utf-8")
).hexdigest()


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Run config must be a mapping.")
    validate_config(config)
    return config


def validate_config(config: Mapping[str, Any]) -> None:
    proxy = deepcopy(dict(config))
    proxy_runpod = deepcopy(dict(proxy["runpod"]))
    proxy_runpod.pop("preflight_terminate_after_hours", None)
    proxy_runpod.pop("scientific_terminate_after_hours", None)
    proxy_runpod["seed_worker_terminate_after_hours"] = 4.0
    proxy_runpod["additional_worker_terminate_after_hours"] = 2.5
    proxy["runpod"] = proxy_runpod
    _BASE.validate_config(proxy)
    if config.get("name") != "pythia-14m-full-minipile-pass-corrected-a4z-ol1":
        raise ValueError("Run 015 config name changed.")
    runpod = config["runpod"]
    if float(runpod.get("preflight_terminate_after_hours", 0.0)) != 1.5:
        raise ValueError("Run 015 preflight requires a 1.5-hour absolute guard.")
    if float(runpod.get("scientific_terminate_after_hours", 0.0)) != 2.5:
        raise ValueError("Run 015 scientific workers require 2.5-hour absolute guards.")
    if "seed_worker_terminate_after_hours" in runpod:
        raise ValueError("Run 015 separates the preflight from scientific workers.")


def run_code_identity() -> dict[str, Any]:
    names = (
        "00_setup_remote.sh", "01_smoke.py", "02_train.py", "03_verify.py",
        "04_monitor.py", "05_remote_preflight.py", "06_build_cache_from_hf.py",
        "_reuse_run004.py", "_reuse_run012.py", "run015_capture.py",
        "run_config.py", "initialization.py", "optimizer_boundary.py",
        "diagnostics.py", "smoke.py", "training.py", "verification.py",
        "../004-2026-08-29-pythia14m-full-pass-l1n/run_config.py",
        "../004-2026-08-29-pythia14m-full-pass-l1n/initialization.py",
        "../004-2026-08-29-pythia14m-full-pass-l1n/optimizer_boundary.py",
        "../004-2026-08-29-pythia14m-full-pass-l1n/diagnostics.py",
        "../004-2026-08-29-pythia14m-full-pass-l1n/smoke.py",
        "../004-2026-08-29-pythia14m-full-pass-l1n/training.py",
        "../004-2026-08-29-pythia14m-full-pass-l1n/06_build_cache_from_hf.py",
        "../012-2026-08-30-pythia14m-full-pass-a4-ol1/run_config.py",
        "../012-2026-08-30-pythia14m-full-pass-a4-ol1/verification.py",
        "../012-2026-08-30-pythia14m-full-pass-a4-ol1/smoke.py",
        "../012-2026-08-30-pythia14m-full-pass-a4-ol1/05_remote_preflight.py",
        "../011-2026-08-30-pythia14m-full-pass-a4z/artifacts/verification.json",
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


condition_specs = _BASE.condition_specs
worker_conditions = _BASE.worker_conditions
resolved_condition_config = _BASE.resolved_condition_config
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

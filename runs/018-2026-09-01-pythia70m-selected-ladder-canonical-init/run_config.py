"""Run 018 identities: frozen Run 017 science plus canonical initialization artifacts."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping

import yaml

from sparsity_research.artifacts import config_sha256

from _reuse_run017 import load_run017_module


RUN_DIR = Path(__file__).resolve().parent
REPO_ROOT = RUN_DIR.parents[1]
DEFAULT_CONFIG = RUN_DIR / "config.yaml"
_FROZEN = load_run017_module("_run018_frozen_run017_config", "run_config.py")
_FROZEN.RUN_DIR = RUN_DIR
_FROZEN.REPO_ROOT = REPO_ROOT
_FROZEN.DEFAULT_CONFIG = DEFAULT_CONFIG
_FROZEN._BASE.RUN_DIR = RUN_DIR
_FROZEN._BASE.REPO_ROOT = REPO_ROOT
_FROZEN._BASE.DEFAULT_CONFIG = DEFAULT_CONFIG

EXPECTED_THRESHOLDS = _FROZEN.EXPECTED_THRESHOLDS
A4_SITES = _FROZEN.A4_SITES
A7_SITES = _FROZEN.A7_SITES
ONE_SIDED_SITES = _FROZEN.ONE_SIDED_SITES
SYMMETRIC_SITES = _FROZEN.SYMMETRIC_SITES
DIAGNOSTIC_SITES = _FROZEN.DIAGNOSTIC_SITES
EXPECTED_CONDITION_IDS = _FROZEN.EXPECTED_CONDITION_IDS
EXPECTED_WORKERS = _FROZEN.EXPECTED_WORKERS
EXPECTED_SENTINEL = _FROZEN.EXPECTED_SENTINEL
EXPECTED_REMAINDER = _FROZEN.EXPECTED_REMAINDER
EXPECTED_SCHEDULE_SHA256 = _FROZEN.EXPECTED_SCHEDULE_SHA256
EXPECTED_INITIAL_PARAMETER_SHA256 = _FROZEN.EXPECTED_INITIAL_PARAMETER_SHA256
EXPECTED_CEILINGS = _FROZEN.EXPECTED_CEILINGS

condition_specs = _FROZEN.condition_specs
worker_conditions = _FROZEN.worker_conditions
resolved_condition_config = _FROZEN.resolved_condition_config
expected_ceiling = _FROZEN.expected_ceiling
build_schedule = _FROZEN.build_schedule
mapping = _FROZEN.mapping
repo_path = _FROZEN.repo_path
load_verified_caches = _FROZEN.load_verified_caches
microbatches_for_step = _FROZEN.microbatches_for_step
require_cuda = _FROZEN.require_cuda
seed_everything = _FROZEN.seed_everything
require_training_cache = _FROZEN.require_training_cache
require_validation_coverage = _FROZEN.require_validation_coverage
parameter_sha256 = _FROZEN.parameter_sha256
inventory_content_sha256 = _FROZEN.inventory_content_sha256
cache_identity = _FROZEN.cache_identity
git_identity = _FROZEN.git_identity
write_json = _FROZEN.write_json


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Run config must be a mapping.")
    validate_config(config)
    return config


def validate_science_config(config: Mapping[str, Any]) -> None:
    """Validate the unchanged Run 017 scientific and execution contract."""

    _FROZEN.validate_config(config)
    model = mapping(config, "model")
    if (
        model.get("initialization_realization") != "hash_pinned_generated_artifact"
        or model.get("released_weights_loaded") is not False
    ):
        raise ValueError("Run 018 must use the approved generated initialization artifact.")


def validate_config(config: Mapping[str, Any]) -> None:
    validate_science_config(config)
    artifact = mapping(config, "initialization_artifact")
    expected = {
        "source_run": "017-2026-09-01-pythia70m-selected-ladder-portable-init",
        "format": "safetensors",
        "model_path": "prelaunch/initialization/pythia70m-seed1234.safetensors",
        "model_bytes": 281_715_344,
        "model_sha256": "024e01975e1a52ead00340afd7a5c3f0b7c2fa0542d9dd5998e648ec14f73501",
        "rng_path": "prelaunch/initialization/pythia70m-seed1234-rng.pt",
        "rng_bytes": 14_823,
        "rng_sha256": "ff839f490cbbbec528181113451802f52c734fb45ae693fc800991bc2be36762",
        "metadata_path": "prelaunch/initialization/metadata.json",
        "parameter_sha256": EXPECTED_INITIAL_PARAMETER_SHA256,
        "tensor_count": 76,
        "tensor_bytes": 281_706_496,
        "cuda_rng_policy": "seed_1234_on_worker_before_artifact_load",
        "locally_generated_random_pretraining_initialization": True,
    }
    if any(artifact.get(key) != value for key, value in expected.items()):
        raise ValueError("The canonical initialization artifact identity changed.")
    if tuple(artifact.get("rng_restore", ())) != ("python", "numpy", "torch_cpu"):
        raise ValueError("The canonical RNG restoration contract changed.")
    for prefix in ("model", "rng"):
        digest = artifact.get(f"{prefix}_sha256")
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError(f"The {prefix} artifact SHA-256 is not pinned.")
        int(digest, 16)
        if int(artifact.get(f"{prefix}_bytes", 0)) <= 0:
            raise ValueError(f"The {prefix} artifact byte count is not pinned.")


def run_code_identity() -> dict[str, Any]:
    names = (
        "00_generate_initialization.py", "01_setup_remote.sh", "02_smoke.py",
        "03_train.py", "04_verify.py", "05_monitor.py", "06_remote_preflight.py",
        "07_teal_posthoc.py", "_reuse_run004.py", "_reuse_run017.py",
        "architecture_config.json", "config.yaml", "diagnostics.py", "initialization.py",
        "initialization_artifact.py", "model_factory.py", "optimizer_boundary.py",
        "prelaunch/initialization/metadata.json",
        "run018_capture.py", "run_config.py", "smoke.py", "teal_posthoc.py",
        "training.py", "verification.py",
        "../017-2026-09-01-pythia70m-selected-ladder-portable-init/_reuse_run004.py",
        "../017-2026-09-01-pythia70m-selected-ladder-portable-init/diagnostics.py",
        "../017-2026-09-01-pythia70m-selected-ladder-portable-init/initialization.py",
        "../017-2026-09-01-pythia70m-selected-ladder-portable-init/optimizer_boundary.py",
        "../017-2026-09-01-pythia70m-selected-ladder-portable-init/run_config.py",
        "../017-2026-09-01-pythia70m-selected-ladder-portable-init/smoke.py",
        "../017-2026-09-01-pythia70m-selected-ladder-portable-init/teal_posthoc.py",
        "../017-2026-09-01-pythia70m-selected-ladder-portable-init/training.py",
        "../017-2026-09-01-pythia70m-selected-ladder-portable-init/verification.py",
        "../017-2026-09-01-pythia70m-selected-ladder-portable-init/05_remote_preflight.py",
        "../004-2026-08-29-pythia14m-full-pass-l1n/run_config.py",
        "../004-2026-08-29-pythia14m-full-pass-l1n/diagnostics.py",
        "../004-2026-08-29-pythia14m-full-pass-l1n/optimizer_boundary.py",
        "../004-2026-08-29-pythia14m-full-pass-l1n/training.py",
        "../../src/sparsity_research/artifacts.py", "../../src/sparsity_research/capture.py",
        "../../src/sparsity_research/ceilings.py", "../../src/sparsity_research/data.py",
        "../../src/sparsity_research/evaluation.py", "../../src/sparsity_research/logical_capture.py",
        "../../src/sparsity_research/metrics.py", "../../src/sparsity_research/optimization.py",
        "../../src/sparsity_research/pressure.py", "../../src/sparsity_research/pythia.py",
        "../../src/sparsity_research/sites.py",
    )
    files = []
    for name in names:
        path = RUN_DIR / name
        payload = path.read_bytes()
        files.append({"path": name, "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()})
    digest = hashlib.sha256()
    for row in files:
        digest.update(row["path"].encode("utf-8"))
        digest.update(b"\0")
        digest.update(row["sha256"].encode("ascii"))
        digest.update(b"\n")
    return {"files": files, "content_sha256": digest.hexdigest()}


def approved_identity() -> dict[str, Any]:
    config = load_config()
    return {"config_sha256": config_sha256(config), "run_code": run_code_identity()}

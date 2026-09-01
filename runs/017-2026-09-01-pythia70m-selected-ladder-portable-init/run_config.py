"""Approved Pythia-70M selected-ladder identities and fail-closed invariants."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import math
from pathlib import Path
from typing import Any, Mapping

import yaml

from sparsity_research.artifacts import config_sha256
from sparsity_research.ceilings import architecture_ceiling
from sparsity_research.pressure import parse_pressure_config
from sparsity_research.sites import resolve_topology_and_gates

from _reuse_run004 import load_run004_module


RUN_DIR = Path(__file__).resolve().parent
REPO_ROOT = RUN_DIR.parents[1]
DEFAULT_CONFIG = RUN_DIR / "config.yaml"
EXPECTED_THRESHOLDS = (0.0, 0.01, 0.05, 0.1, 0.5)
A4_SITES = ("a", "m", "h", "z")
A7_SITES = ("a", "m", "h", "q_post", "k_post", "v", "z")
ONE_SIDED_SITES = ("a", "m", "h", "z")
SYMMETRIC_SITES = ("q_post", "k_post", "v")
DIAGNOSTIC_SITES = ("a", "m", "h", "q_post", "k_post", "v", "z", "attention_output")
EXPECTED_CONDITION_IDS = (
    "a0-gelu",
    "a1h-relu",
    "a4-ol1-kappa-0",
    "a4-ol1-kappa-0p01",
    "a4-ol1-kappa-0p05",
    "a4-ol1-kappa-0p1",
    "a4-ol1-kappa-0p5",
    "a7-ol1-kappa-0",
    "a7-ol1-kappa-0p01",
    "a7-ol1-kappa-0p05",
    "a7-ol1-kappa-0p1",
    "a7-ol1-kappa-0p5",
)
EXPECTED_WORKERS = {condition_id: (condition_id,) for condition_id in EXPECTED_CONDITION_IDS}
EXPECTED_SENTINEL = (
    "a0-gelu",
    "a1h-relu",
    "a4-ol1-kappa-0",
    "a7-ol1-kappa-0",
)
EXPECTED_REMAINDER = tuple(
    condition_id for condition_id in EXPECTED_CONDITION_IDS if condition_id not in EXPECTED_SENTINEL
)
EXPECTED_SCHEDULE_SHA256 = "d17a6c0c0d4aacff4b477e6d576f511c12c04ebbc37468f08e6fe61ff1c6ad8e"
EXPECTED_INITIAL_PARAMETER_SHA256 = "e8b8d8e48880f8ff25e421ed29b04a81eb417300f2b4a01a8c4d56f2591a1062"
EXPECTED_CEILINGS = {
    "A0": (0, 104_293_466_112),
    "A1-H": (12_884_901_888, 104_293_466_112),
    "A4-Z": (38_654_705_664, 104_293_466_112),
    "A7-Z-POST": (51_545_899_008, 104_293_466_112),
}


_BASE = load_run004_module("_run017_frozen_run004_config", "run_config.py")
_BASE.RUN_DIR = RUN_DIR
_BASE.REPO_ROOT = REPO_ROOT
_BASE.DEFAULT_CONFIG = DEFAULT_CONFIG


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Run config must be a mapping.")
    validate_config(config)
    return config


def site_gates(topology_id: str, kappa: float) -> dict[str, dict[str, Any]]:
    topology, _ = resolve_topology_and_gates(
        topology_id,
        None,
        {
            site: {
                "operator": (
                    "one_sided_threshold" if site in ONE_SIDED_SITES else "symmetric_threshold"
                ),
                "kappa": float(kappa),
            }
            for site in (A4_SITES if topology_id == "A4-Z" else A7_SITES)
        },
    )
    return {
        site: {
            "operator": (
                "one_sided_threshold" if site in ONE_SIDED_SITES else "symmetric_threshold"
            ),
            "kappa": float(kappa),
        }
        for site in topology.active_sites
    }


def condition_specs(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    thresholds = tuple(float(value) for value in mapping(config, "conditions")["gate_thresholds"])
    rows = [
        {
            "id": "a0-gelu",
            "order": 1,
            "step": "A0",
            "topology_id": "A0",
            "active_sites": [],
            "gate_threshold": None,
            "pressure_method": "none",
            "pressure_sites": [],
            "pressure_weight": 0.0,
            "step_budget": None,
            "label": "GeLU control",
            "is_control": True,
        },
        {
            "id": "a1h-relu",
            "order": 2,
            "step": "A1-H",
            "topology_id": "A1-H",
            "active_sites": ["h"],
            "gate_threshold": None,
            "pressure_method": "none",
            "pressure_sites": [],
            "pressure_weight": 0.0,
            "step_budget": None,
            "label": "ReLU control",
            "is_control": True,
        },
    ]
    order = 3
    for step, topology_id, active_sites in (
        ("A4-OL1", "A4-Z", A4_SITES),
        ("A7-OL1", "A7-Z-POST", A7_SITES),
    ):
        prefix = "a4" if topology_id == "A4-Z" else "a7"
        for threshold in thresholds:
            token = f"{threshold:g}".replace(".", "p")
            rows.append(
                {
                    "id": f"{prefix}-ol1-kappa-{token}",
                    "order": order,
                    "step": step,
                    "topology_id": topology_id,
                    "active_sites": list(active_sites),
                    "gate_threshold": threshold,
                    "pressure_method": "orthogonal_l1",
                    "pressure_sites": list(active_sites),
                    "pressure_weight": 1.0,
                    "step_budget": 1.0,
                    "label": f"kappa={threshold:g}, lambda=1",
                    "is_control": False,
                }
            )
            order += 1
    return rows


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
    topology_id = str(condition["topology_id"])
    threshold = condition.get("gate_threshold")
    if topology_id == "A0":
        uniform_gate, gates = None, None
    elif topology_id == "A1-H":
        uniform_gate, gates = {"operator": "relu"}, None
    else:
        uniform_gate, gates = None, site_gates(topology_id, float(threshold))
    topology, realized_gates = resolve_topology_and_gates(topology_id, uniform_gate, gates)
    if tuple(topology.active_sites) != tuple(condition["active_sites"]):
        raise RuntimeError("Resolved topology sites differ from the condition identity.")
    resolved["model"].update(
        topology_id=topology_id,
        site_gate=uniform_gate,
        site_gates=(None if gates is None else realized_gates),
        pressure_sites=list(condition["pressure_sites"]),
    )
    pressure = {
        "method": str(condition["pressure_method"]),
        "sites": list(condition["pressure_sites"]),
        "weight": float(condition["pressure_weight"]),
        "step_budget": condition["step_budget"],
        "eps": 1e-12,
    }
    parsed = parse_pressure_config(pressure)
    if parsed.enabled != (not bool(condition["is_control"])):
        raise RuntimeError("Resolved pressure state differs from the condition identity.")
    resolved["activation_pressure"] = pressure
    return resolved


def expected_ceiling(topology_id: str) -> dict[str, Any]:
    result = architecture_ceiling(
        topology_id,
        layers=6,
        hidden_size=512,
        ffn_size=2048,
        sequence_length=2048,
        vocabulary_size=50304,
    )
    expected = EXPECTED_CEILINGS[topology_id]
    observed = (int(result["reachable_product_count"]), int(result["model_product_count"]))
    if observed != expected:
        raise RuntimeError(f"Pythia-70M R_model_max arithmetic changed: {observed} != {expected}")
    return result


def build_schedule(config: Mapping[str, Any], train_metadata: Mapping[str, Any], *, np: Any):
    result = _BASE.build_schedule(config, train_metadata, np=np)
    if result[1] != EXPECTED_SCHEDULE_SHA256:
        raise RuntimeError("Run 017 full-pass schedule identity changed.")
    return result


def validate_config(config: Mapping[str, Any]) -> None:
    model = mapping(config, "model")
    expected_model = {
        "architecture": "EleutherAI/pythia-70m-deduped",
        "revision": "e93a9faa9c77e5d09219f6c868bfc7a1bd65593c",
        "initialization": "random",
        "initialization_method": "small_init",
        "output_layer_initialization_method": "wang_init",
        "attention_implementation": "sdpa_flash",
        "execution_engine": "transformers_recipe_mapping",
    }
    if any(model.get(key) != value for key, value in expected_model.items()):
        raise ValueError("The pinned random Pythia-70M model identity changed.")

    recipe = mapping(config, "recipe")
    if (
        recipe.get("exact_framework_reproduction") is not False
        or recipe.get("optimizer_mapping") != "pytorch_fused_adamw"
        or tuple(recipe.get("weight_decay_exclusions", ())) != ("bias", "layer_norm")
        or recipe.get("lr_schedule_semantics") != "gpt_neox_v1_pre_step"
    ):
        raise ValueError("The approved GPT-NeoX-to-Transformers recipe mapping changed.")
    if dict(mapping(config, "runtime")) != {
        "python": "3.12",
        "torch": "2.11.0",
        "transformers": "5.12.1",
        "cuda_runtime": "12.8",
    }:
        raise ValueError("The pinned runtime changed.")

    data = mapping(config, "data")
    if (
        data.get("dataset") != "JeanKaddour/minipile"
        or data.get("revision") != "18ad1b0c701eaa0de03d3cecfdd769cbc70ffbd0"
        or data.get("tokenizer") != "EleutherAI/pythia-14m-deduped"
        or data.get("tokenizer_revision") != "7386d9a4ae45aef494a6e704910394def3037fc5"
        or data.get("append_eos") is not True
        or int(data.get("sequence_length", 0)) != 2048
    ):
        raise ValueError("The hash-verified MiniPile/tokenizer contract changed.")
    seeds = mapping(config, "seeds")
    if int(seeds.get("model", -1)) != 1234 or int(seeds.get("data_order", -1)) != 1234:
        raise ValueError("The approved model/data seed changed.")

    conditions = mapping(config, "conditions")
    if tuple(conditions.get("selected_steps", ())) != ("A0", "A1-H", "A4-OL1", "A7-OL1"):
        raise ValueError("The selected ladder changed.")
    if tuple(float(value) for value in conditions.get("gate_thresholds", ())) != EXPECTED_THRESHOLDS:
        raise ValueError("The approved kappa grid changed.")
    if (
        conditions.get("pressure_method") != "orthogonal_l1"
        or float(conditions.get("pressure_weight", math.nan)) != 1.0
        or float(conditions.get("step_budget", math.nan)) != 1.0
    ):
        raise ValueError("The approved OL1 method, lambda, or trust budget changed.")
    rows = condition_specs(config)
    if tuple(row["id"] for row in rows) != EXPECTED_CONDITION_IDS:
        raise ValueError("The 12-condition identity/order changed.")
    for topology_id in EXPECTED_CEILINGS:
        expected_ceiling(topology_id)

    training = mapping(config, "training")
    expected_training = {
        "max_steps": 712,
        "global_batch_size": 1024,
        "micro_batch_size": 4,
        "gradient_accumulation_steps": 256,
        "optimizer": "adamw",
        "precision": "float16_dynamic",
        "parameter_dtype": "float32",
        "activation_checkpointing": False,
        "device": "cuda",
        "fp16_ol1_overflow_policy": "skip_entire_boundary",
    }
    if any(training.get(key) != value for key, value in expected_training.items()):
        raise ValueError("The proposed 70M training decomposition or precision changed.")

    model = mapping(config, "model")
    if model.get("initialization_device") != "cpu" or model.get("training_device") != "cuda":
        raise ValueError("Run 017 must initialize on CPU before transferring to CUDA.")
    if int(training["micro_batch_size"]) * int(training["gradient_accumulation_steps"]) != 1024:
        raise ValueError("Microbatch times accumulation must equal the global batch.")
    expected_scalars = {
        "peak_learning_rate": 1e-3,
        "minimum_learning_rate": 1e-4,
        "adamw_eps": 1e-8,
        "weight_decay": 0.1,
        "gradient_clip_norm": 1.0,
        "warmup_fraction": 0.01,
        "hidden_dropout": 0.0,
        "attention_dropout": 0.0,
        "minimum_loss_scale": 1.0,
    }
    if any(float(training.get(key, math.nan)) != value for key, value in expected_scalars.items()):
        raise ValueError("A locked Pythia training scalar changed.")
    if tuple(float(value) for value in training.get("adamw_betas", ())) != (0.9, 0.95):
        raise ValueError("AdamW betas changed.")
    if (
        int(training.get("initial_loss_scale_power", -1)) != 12
        or int(training.get("loss_scale_window", -1)) != 1000
        or int(training.get("loss_scale_hysteresis", -1)) != 2
    ):
        raise ValueError("Dynamic FP16 scaling changed.")

    validation = mapping(config, "validation")
    expected_validation = {
        "documents": 500,
        "complete_sequences": 338,
        "input_tokens": 692_224,
        "excluded_tail_tokens": 1_444,
        "batch_size": 4,
    }
    if any(int(validation.get(key, -1)) != value for key, value in expected_validation.items()):
        raise ValueError("Complete validation coverage or batch size changed.")

    diagnostics = mapping(config, "diagnostics")
    if tuple(diagnostics.get("activation_sites", ())) != DIAGNOSTIC_SITES:
        raise ValueError("The approved diagnostic sites changed.")
    if tuple(float(value) for value in diagnostics.get("near_zero_thresholds", ())) != (0.0, 0.001, 0.01):
        raise ValueError("The near-zero thresholds changed.")
    required_true = (
        "activation_statistics", "weight_statistics", "weight_statistics_include_bias_and_norm",
        "gradient_interaction", "ol1_adaptive_directions", "logical_products",
    )
    if any(diagnostics.get(key) is not True for key in required_true):
        raise ValueError("An approved diagnostic was disabled.")
    if int(diagnostics.get("logical_product_batch_size", 0)) != 1:
        raise ValueError("Logical-product evaluation must remain batch one.")

    teal = mapping(config, "teal_posthoc")
    if (
        tuple(teal.get("condition_ids", ())) != ("a0-gelu", "a1h-relu")
        or tuple(teal.get("sites", ())) != A4_SITES
        or tuple(float(value) for value in teal.get("target_sparsities", ())) != tuple(i / 10 for i in range(10))
        or int(teal.get("calibration_blocks", 0)) != 10
        or teal.get("calibration_split") != "train"
        or teal.get("calibration_selection") != "first_complete_source_order_blocks"
        or int(teal.get("evaluation_batch_size", 0)) != 1
        or float(teal.get("zero_threshold_loss_tolerance", math.nan)) != 5e-4
    ):
        raise ValueError("The Analysis 005/006 TEAL protocol changed.")

    checkpoints = mapping(config, "checkpoints")
    if (
        tuple(int(value) for value in checkpoints.get("model_steps", ())) != (712,)
        or tuple(int(value) for value in checkpoints.get("optimizer_steps", ())) != (712,)
        or checkpoints.get("retain_final") is not True
        or checkpoints.get("retain_initial_weights") is not False
    ):
        raise ValueError("Run 017 must retain only the complete final recovery checkpoint.")

    runpod = mapping(config, "runpod")
    assignments = {
        str(key): tuple(str(item) for item in value)
        for key, value in mapping(runpod, "worker_assignments").items()
    }
    waves = mapping(runpod, "launch_waves")
    if assignments != EXPECTED_WORKERS or int(runpod.get("maximum_parallel_pods", 0)) != 8:
        raise ValueError("Run 017 requires one independently assigned condition per GPU.")
    if tuple(waves.get("sentinel", ())) != EXPECTED_SENTINEL or tuple(waves.get("remainder", ())) != EXPECTED_REMAINDER:
        raise ValueError("The approved four-condition sentinel/eight-condition remainder plan changed.")
    if set(waves["sentinel"]).intersection(waves["remainder"]) or set(waves["sentinel"]) | set(waves["remainder"]) != set(EXPECTED_CONDITION_IDS):
        raise ValueError("Launch waves must partition all 12 conditions exactly once.")
    if (
        runpod.get("candidate_gpu_type") != "NVIDIA H200"
        or int(runpod.get("candidate_gpu_memory_gb", 0)) != 141
        or runpod.get("cloud_type") != "SECURE"
        or runpod.get("parallelism") != "one_condition_per_gpu"
        or tuple(runpod.get("candidate_data_centers", ()))
        != ("US-NC-1", "US-CA-2", "US-GA-2", "AP-JP-1", "EUR-IS-5")
        or runpod.get("storage_strategy") != "per_pod_volume_seeded_from_hash_verified_cache"
        or runpod.get("retained_network_volume_used") is not False
        or float(runpod.get("preflight_terminate_after_hours", 0.0)) != 1.0
        or float(runpod.get("scientific_terminate_after_hours", 0.0)) != 6.5
    ):
        raise ValueError("The approved H200 portability execution envelope changed.")
    if runpod.get("image") != "runpod/pytorch@sha256:0a360022e8de4375af99430f84e8b38951acc397252163a37ceac7204d01be35":
        raise ValueError("The pinned RunPod image digest changed.")

    artifacts = mapping(config, "artifacts")
    if dict(artifacts) != {
        "save_final_checkpoint": True,
        "save_optimizer_at_declared_steps": True,
        "retain_predictions": False,
        "transfer_final_checkpoint_only": True,
    }:
        raise ValueError("The approved artifact retention contract changed.")


def run_code_identity() -> dict[str, Any]:
    names = (
        "00_setup_remote.sh", "01_smoke.py", "02_train.py", "03_verify.py",
        "04_monitor.py", "05_remote_preflight.py", "06_teal_posthoc.py",
        "_reuse_run004.py", "architecture_config.json", "config.yaml", "diagnostics.py",
        "initialization.py", "model_factory.py", "optimizer_boundary.py", "run017_capture.py", "run_config.py", "smoke.py",
        "teal_posthoc.py", "training.py", "verification.py",
        "../004-2026-08-29-pythia14m-full-pass-l1n/run_config.py",
        "../004-2026-08-29-pythia14m-full-pass-l1n/diagnostics.py",
        "../004-2026-08-29-pythia14m-full-pass-l1n/optimizer_boundary.py",
        "../004-2026-08-29-pythia14m-full-pass-l1n/training.py",
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


mapping = _BASE.mapping
repo_path = _BASE.repo_path
load_verified_caches = _BASE.load_verified_caches
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

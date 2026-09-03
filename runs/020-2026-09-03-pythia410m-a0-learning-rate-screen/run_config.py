"""Pythia-410M A0 learning-rate screen identities and fail-closed invariants."""

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
A4_SITES = ("a", "m", "h", "z")
A7_SITES = ("a", "m", "h", "q_post", "k_post", "v", "z")
ONE_SIDED_SITES = ("a", "m", "h", "z")
SYMMETRIC_SITES = ("q_post", "k_post", "v")
DIAGNOSTIC_SITES = ("a", "m", "h", "q_post", "k_post", "v", "z", "attention_output")
EXPECTED_CONDITION_IDS = (
    "a0-lr-6e-4",
    "a0-lr-1e-3",
)
EXPECTED_WORKERS = {condition_id: (condition_id,) for condition_id in EXPECTED_CONDITION_IDS}
EXPECTED_PEAK_LEARNING_RATES = (6e-4, 1e-3)
EXPECTED_SCHEDULE_SHA256 = "d17a6c0c0d4aacff4b477e6d576f511c12c04ebbc37468f08e6fe61ff1c6ad8e"
EXPECTED_INITIAL_PARAMETER_SHA256 = "76217bf2ef13de2377c7750515a559cb71f574c274476e08408a5f7749ea9cff"
EXPECTED_CEILINGS = {
    "A0": (0, 827_099_971_584),
}


_BASE = load_run004_module("_run020_frozen_run004_config", "run_config.py")
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
    peaks = tuple(
        float(value)
        for value in mapping(config, "conditions")["required_peak_learning_rates"]
    )
    ratio = float(mapping(config, "conditions")["minimum_learning_rate_ratio"])
    return [
        {
            "id": condition_id,
            "order": order,
            "step": "A0",
            "topology_id": "A0",
            "active_sites": [],
            "gate_threshold": None,
            "pressure_method": "none",
            "pressure_sites": [],
            "pressure_weight": 0.0,
            "step_budget": None,
            "peak_learning_rate": peak,
            "minimum_learning_rate": peak * ratio,
            "label": f"A0, peak LR={peak:g}",
            "is_control": True,
        }
        for order, (condition_id, peak) in enumerate(
            zip(EXPECTED_CONDITION_IDS, peaks, strict=True), start=1
        )
    ]


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
    if topology_id != "A0":
        raise RuntimeError("Run 020 contains only the A0 topology.")
    uniform_gate, gates = None, None
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
    resolved["training"]["peak_learning_rate"] = float(condition["peak_learning_rate"])
    resolved["training"]["minimum_learning_rate"] = float(condition["minimum_learning_rate"])
    return resolved


def expected_ceiling(topology_id: str) -> dict[str, Any]:
    result = architecture_ceiling(
        topology_id,
        layers=24,
        hidden_size=1024,
        ffn_size=4096,
        sequence_length=2048,
        vocabulary_size=50304,
    )
    expected = EXPECTED_CEILINGS[topology_id]
    observed = (int(result["reachable_product_count"]), int(result["model_product_count"]))
    if observed != expected:
        raise RuntimeError(f"Pythia-410M R_model_max arithmetic changed: {observed} != {expected}")
    return result


def build_schedule(config: Mapping[str, Any], train_metadata: Mapping[str, Any], *, np: Any):
    result = _BASE.build_schedule(config, train_metadata, np=np)
    if result[1] != EXPECTED_SCHEDULE_SHA256:
        raise RuntimeError("Run 020 full-pass schedule identity changed.")
    return result


def validate_science_config(config: Mapping[str, Any]) -> None:
    """Validate the approved A0 learning-rate screen before artifact access."""
    model = mapping(config, "model")
    expected_model = {
        "architecture": "EleutherAI/pythia-410m-deduped",
        "revision": "b5e8535141902c0e985cea61fd02afe7fe86af32",
        "initialization": "random",
        "initialization_method": "small_init",
        "output_layer_initialization_method": "wang_init",
        "attention_implementation": "sdpa_flash",
        "execution_engine": "transformers_recipe_mapping",
    }
    if any(model.get(key) != value for key, value in expected_model.items()):
        raise ValueError("The pinned random Pythia-410M model identity changed.")
    if (
        model.get("initialization_realization") != "hash_pinned_generated_artifact"
        or model.get("released_weights_loaded") is not False
    ):
        raise ValueError("Run 020 must reuse the pinned canonical random initialization artifact.")

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
    if conditions.get("selected_step") != "A0":
        raise ValueError("Run 020 is restricted to A0.")
    if (
        tuple(float(value) for value in conditions.get("required_peak_learning_rates", ()))
        != EXPECTED_PEAK_LEARNING_RATES
        or float(conditions.get("minimum_learning_rate_ratio", math.nan)) != 0.1
    ):
        raise ValueError("The approved A0 learning-rate screen changed.")
    rows = condition_specs(config)
    if tuple(row["id"] for row in rows) != EXPECTED_CONDITION_IDS:
        raise ValueError("The two-condition identity/order changed.")
    if tuple(float(row["peak_learning_rate"]) for row in rows) != EXPECTED_PEAK_LEARNING_RATES:
        raise ValueError("The resolved learning-rate arms changed.")
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
        raise ValueError("The proposed 410M training decomposition or precision changed.")

    model = mapping(config, "model")
    if model.get("initialization_device") != "cpu" or model.get("training_device") != "cuda":
        raise ValueError("Run 020 must initialize on CPU before transferring to CUDA.")
    if int(training["micro_batch_size"]) * int(training["gradient_accumulation_steps"]) != 1024:
        raise ValueError("Microbatch times accumulation must equal the global batch.")
    if training.get("peak_learning_rate") is not None or training.get("minimum_learning_rate") is not None:
        raise ValueError("Top-level learning rates must be resolved from the condition identity.")
    expected_scalars = {
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
    if (
        validation.get("at_step_one") is not True
        or validation.get("at_final_step") is not True
        or validation.get("canonical_endpoint_loss") != "logical_product_diagnostic_eager"
    ):
        raise ValueError("The paired eager endpoint-loss contract changed.")

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
    if diagnostics.get("activation_attention_implementation") != "eager":
        raise ValueError("Activation and logical diagnostics must both use eager attention.")

    teal = mapping(config, "teal_posthoc")
    if (
        teal.get("eligibility") != "selected_new_condition_only"
        or tuple(teal.get("condition_ids", ())) != EXPECTED_CONDITION_IDS
        or tuple(teal.get("clipping_sites", ())) != A4_SITES
        or tuple(teal.get("diagnostic_sites", ())) != DIAGNOSTIC_SITES
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
        raise ValueError("Run 020 must retain only the complete final recovery checkpoint.")

    runpod = mapping(config, "runpod")
    assignments = {
        str(key): tuple(str(item) for item in value)
        for key, value in mapping(runpod, "worker_assignments").items()
    }
    waves = mapping(runpod, "launch_waves")
    if assignments != EXPECTED_WORKERS or int(runpod.get("maximum_parallel_pods", 0)) != 2:
        raise ValueError("Run 020 requires one independently assigned LR arm per GPU.")
    if tuple(waves.get("primary", ())) != EXPECTED_CONDITION_IDS:
        raise ValueError("The planned two-arm launch wave changed.")
    selected_priority = (
        "NVIDIA RTX PRO 6000 Blackwell Server Edition",
        "NVIDIA A100-SXM4-80GB",
        "NVIDIA H100 80GB HBM3",
        "NVIDIA H200",
    )
    if (
        runpod.get("selected_gpu_strategy") != "RTX_PRO_6000_PREFERRED_WITH_APPROVED_FALLBACKS"
        or tuple(runpod.get("selected_gpu_type_priority", ())) != selected_priority
        or int(runpod.get("minimum_selected_gpu_memory_gb", 0)) != 80
        or runpod.get("selected_cloud_type")
        != "COMMUNITY_PREFERRED_SECURE_FALLBACK_PER_SKU"
        or runpod.get("mixed_gpu_execution_approved") is not True
        or runpod.get("parallelism") != "one_condition_per_gpu"
        or runpod.get("storage_strategy")
        != "per_pod_volume_seeded_from_one_hash_verified_payload"
        or runpod.get("retained_network_volume_used") is not False
        or int(runpod.get("monitoring_interval_minutes", 0)) != 30
        or int(runpod.get("stale_event_minutes", 0)) != 60
    ):
        raise ValueError("The selected mixed-GPU RunPod execution envelope changed.")
    expected_guards = {
        "NVIDIA RTX PRO 6000 Blackwell Server Edition": 10.0,
        "NVIDIA A100-SXM4-80GB": 12.0,
        "NVIDIA H100 80GB HBM3": 10.0,
        "NVIDIA H200": 8.0,
    }
    if dict(mapping(runpod, "terminate_after_hours_by_gpu")) != expected_guards:
        raise ValueError("The per-SKU automatic-termination guards changed.")
    if runpod.get("image") != "runpod/pytorch@sha256:0a360022e8de4375af99430f84e8b38951acc397252163a37ceac7204d01be35":
        raise ValueError("The pinned RunPod image digest changed.")

    calibration = mapping(config, "calibration_reuse")
    if dict(calibration) != {
        "source": "runs/019-2026-09-01-pythia410m-selected-ladder-canonical-init/prelaunch/calibration/RESULTS.md",
        "scientific_workload_reference": "a0-gelu",
        "exact_architecture_data_batch_and_diagnostics_match": True,
        "new_calibration_required": False,
    }:
        raise ValueError("The approved Run 019 A0 calibration reuse changed.")

    selection = mapping(config, "selection")
    if dict(selection) != {
        "metric": "mean_task_loss_final_64_optimizer_boundaries",
        "use_validation_for_selection": False,
        "require_finite_all_boundaries": True,
        "require_no_skipped_or_overflow_boundaries": True,
        "if_1e3_is_stable_best": "defer_and_propose_fresh_a0_1p5e3_full_pass",
        "otherwise": "select_stable_lowest_metric",
    }:
        raise ValueError("The predeclared learning-rate selection rule changed.")

    artifacts = mapping(config, "artifacts")
    if dict(artifacts) != {
        "save_final_checkpoint": True,
        "save_optimizer_at_declared_steps": True,
        "retain_predictions": False,
        "transfer_final_checkpoint_only": True,
    }:
        raise ValueError("The approved artifact retention contract changed.")


def validate_config(config: Mapping[str, Any]) -> None:
    """Validate science, baseline provenance, and the reused initialization."""

    validate_science_config(config)
    artifact = mapping(config, "initialization_artifact")
    expected = {
        "provenance_run": "runs/019-2026-09-01-pythia410m-selected-ladder-canonical-init",
        "format": "safetensors",
        "model_path": "../019-2026-09-01-pythia410m-selected-ladder-canonical-init/prelaunch/initialization/pythia410m-seed1234.safetensors",
        "model_bytes": 1_621_370_392,
        "model_sha256": "edba25fa7535bef6469228afe7d60db741a1a4ecd6c0c4362989e0b45a8cac51",
        "rng_path": "../019-2026-09-01-pythia410m-selected-ladder-canonical-init/prelaunch/initialization/pythia410m-seed1234-rng.pt",
        "rng_bytes": 14_830,
        "rng_sha256": "38c9f36a2e259018c183039771c5732d8aa5a5f63b4363d9e6403e6683a9ef43",
        "metadata_path": "../019-2026-09-01-pythia410m-selected-ladder-canonical-init/prelaunch/initialization/metadata.json",
        "metadata_bytes": 2_320,
        "metadata_sha256": "b909c5c578ee264127097b7c2f226cabf982de45b9033dc2d9f8e413ae174523",
        "parameter_sha256": EXPECTED_INITIAL_PARAMETER_SHA256,
        "tensor_count": 292,
        "tensor_bytes": 1_621_336_064,
        "cuda_rng_policy": "seed_1234_on_worker_before_artifact_load",
        "locally_generated_random_pretraining_initialization": True,
    }
    if any(artifact.get(key) != value for key, value in expected.items()):
        raise ValueError("The canonical 410M initialization identity changed.")
    if tuple(artifact.get("rng_restore", ())) != ("python", "numpy", "torch_cpu"):
        raise ValueError("The canonical RNG restoration contract changed.")
    if EXPECTED_INITIAL_PARAMETER_SHA256 == "PENDING_LOCAL_GENERATION":
        raise ValueError("The canonical initialization has not been generated and pinned.")
    for prefix in ("model", "rng", "metadata"):
        digest = artifact.get(f"{prefix}_sha256")
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError(f"The {prefix} artifact SHA-256 is not pinned.")
        int(digest, 16)
        if int(artifact.get(f"{prefix}_bytes", 0)) <= 0:
            raise ValueError(f"The {prefix} artifact byte count is not pinned.")

    baseline = mapping(config, "baseline")
    expected_baseline = {
        "source_run": "runs/019-2026-09-01-pythia410m-selected-ladder-canonical-init",
        "condition_id": "a0-gelu",
        "attempt_id": "001-20260902-141527-bcb97fb1",
        "peak_learning_rate": 3e-4,
        "minimum_learning_rate": 3e-5,
        "manifest_sha256": "320aeac7ecc90970ee92be508bf172f64b1e06b60b87e2e8eb5c6537ae3d3758",
        "metrics_sha256": "164b5fde0314d79e6d27ae0fec9fa2edab25e2be74bcb3ddb92dba4411921ba6",
        "events_sha256": "1f59fe88f6b2c051f85fbe74c26078c5c616dcc0dfabf6fe5a207113043891bd",
        "transfer_inventory_sha256": "8935e23cc163eb55232ab3aafbb75ae34adb4aa5a1ff6d1fdbd85bc33d0bab64",
        "config_sha256": "55914254e6e65082fb9efbee01c254590065f214efcbca7034c0335c54df35e3",
        "run_code_sha256": "9fb300223fb57058c42e4030dced0570ce5aadd206db42eebdbeec1d59c4d652",
        "final_checkpoint_content_sha256": "740263f4169a66c94d2117fab7cf9bfd8dd4e102a84d329660766c08eccf8b97",
    }
    if dict(baseline) != expected_baseline:
        raise ValueError("The pinned Run 019 A0 baseline provenance changed.")


def run_code_identity() -> dict[str, Any]:
    names = (
        "01_setup_remote.sh", "02_train.py", "03_verify.py", "04_monitor.py",
        "05_select.py", "06_teal_posthoc.py", "07_smoke.py", "08_verify_initialization.py",
        "_reuse_run004.py", "architecture_config.json", "config.yaml", "diagnostics.py",
        "initialization.py", "initialization_artifact.py", "model_factory.py",
        "optimizer_boundary.py", "run_config.py", "selection.py",
        "smoke.py", "teal_posthoc.py", "training.py", "verification.py",
        "launch-control/prelaunch/finalize_worker.sh",
        "launch-control/prelaunch/build_input_payload.sh",
        "launch-control/prelaunch/guard_runpod_deadline.ps1",
        "launch-control/prelaunch/preflight_worker.sh",
        "launch-control/prelaunch/prepare_worker.sh",
        "launch-control/prelaunch/start_worker.sh",
        "launch-control/prelaunch/verify_worker.sh",
        "../019-2026-09-01-pythia410m-selected-ladder-canonical-init/prelaunch/initialization/metadata.json",
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
        source = path.read_bytes()
        payload = source.replace(b"\r\n", b"\n")
        files.append(
            {
                "path": name,
                "source_bytes": len(source),
                "canonical_lf_bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    digest = hashlib.sha256()
    for row in files:
        digest.update(row["path"].encode("utf-8"))
        digest.update(b"\0")
        digest.update(row["sha256"].encode("ascii"))
        digest.update(b"\n")
    return {
        "newline_policy": "text inputs canonicalized from CRLF to LF before hashing",
        "files": files,
        "content_sha256": digest.hexdigest(),
    }


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

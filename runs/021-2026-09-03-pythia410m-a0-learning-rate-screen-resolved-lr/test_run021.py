from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest

import run_config
import selection
import teal_posthoc
import training
import verification
from optimizer_boundary import recipe_learning_rate


def test_two_required_full_pass_lr_arms_are_exact_and_resolved():
    config = run_config.load_config()
    rows = run_config.condition_specs(config)
    assert tuple(row["id"] for row in rows) == run_config.EXPECTED_CONDITION_IDS
    assert [row["order"] for row in rows] == [1, 2]
    assert [row["peak_learning_rate"] for row in rows] == [6e-4, 1e-3]
    assert [row["minimum_learning_rate"] for row in rows] == pytest.approx([6e-5, 1e-4])
    for row in rows:
        resolved = run_config.resolved_condition_config(config, row)
        assert resolved["model"]["topology_id"] == "A0"
        assert resolved["model"]["site_gate"] is None
        assert resolved["model"]["site_gates"] is None
        assert resolved["model"]["pressure_sites"] == []
        assert resolved["activation_pressure"]["method"] == "none"
        assert resolved["training"]["peak_learning_rate"] == row["peak_learning_rate"]
        assert resolved["training"]["minimum_learning_rate"] == row["minimum_learning_rate"]


def test_schedule_data_and_initialization_identity_match_run019():
    config = run_config.load_config()
    metadata = json.loads(
        run_config.repo_path(config["data"]["training_metadata"]).read_text(encoding="utf-8")
    )
    _, schedule_sha256, schedule = run_config.build_schedule(config, metadata, np=np)
    assert schedule_sha256 == run_config.EXPECTED_SCHEDULE_SHA256
    assert schedule["max_steps"] == 712
    assert schedule["scheduled_blocks"] == 729_088
    assert schedule["wrapped_blocks"] == 714
    assert 712 * 1024 * 2048 == 1_493_172_224
    artifact = config["initialization_artifact"]
    assert artifact["parameter_sha256"] == run_config.EXPECTED_INITIAL_PARAMETER_SHA256
    for prefix in ("model", "rng", "metadata"):
        path = run_config.RUN_DIR / artifact[f"{prefix}_path"]
        assert path.is_file()
        assert path.stat().st_size == artifact[f"{prefix}_bytes"]
    metadata_artifact = json.loads(
        (run_config.RUN_DIR / artifact["metadata_path"]).read_text(encoding="utf-8")
    )
    assert metadata_artifact["parameter_sha256"] == run_config.EXPECTED_INITIAL_PARAMETER_SHA256
    assert metadata_artifact["released_weights_loaded"] is False


def test_baseline_small_artifacts_are_content_pinned():
    config = run_config.load_config()
    baseline = config["baseline"]
    root = (
        run_config.REPO_ROOT
        / baseline["source_run"]
        / "artifacts"
        / "attempts"
        / baseline["attempt_id"]
    )
    for name, key in (
        ("manifest.json", "manifest_sha256"),
        ("metrics.json", "metrics_sha256"),
        ("events.jsonl", "events_sha256"),
        ("transfer_inventory.json", "transfer_inventory_sha256"),
    ):
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == baseline[key]
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["initial_parameter_sha256"] == run_config.EXPECTED_INITIAL_PARAMETER_SHA256
    assert manifest["training_schedule_hash"] == run_config.EXPECTED_SCHEDULE_SHA256
    assert manifest["gradient_overflow_steps"] == []


def test_learning_rate_schedules_differ_only_by_scale():
    config = run_config.load_config()
    rows = run_config.condition_specs(config)
    values = []
    for row in rows:
        values.append(
            [
                recipe_learning_rate(
                    step,
                    peak=row["peak_learning_rate"],
                    max_steps=712,
                    warmup_fraction=0.01,
                    minimum=row["minimum_learning_rate"],
                )
                for step in range(1, 713)
            ]
        )
    assert values[0][0] == values[1][0] == 0.0
    assert values[0][-1] == pytest.approx(6e-5)
    assert values[1][-1] == pytest.approx(1e-4)
    for low, high in zip(values[0][1:], values[1][1:], strict=True):
        assert high / low == pytest.approx(5 / 3)


def test_config_fails_closed_on_scientific_or_selection_change():
    config = run_config.load_config()
    changes = []
    changed = deepcopy(config)
    changed["conditions"]["required_peak_learning_rates"] = [4e-4, 1e-3]
    changes.append(changed)
    changed = deepcopy(config)
    changed["training"]["max_steps"] = 711
    changes.append(changed)
    changed = deepcopy(config)
    changed["baseline"]["events_sha256"] = "0" * 64
    changes.append(changed)
    changed = deepcopy(config)
    changed["selection"]["use_validation_for_selection"] = True
    changes.append(changed)
    changed = deepcopy(config)
    changed["runpod"]["maximum_parallel_pods"] = 3
    changes.append(changed)
    for value in changes:
        with pytest.raises(ValueError):
            run_config.validate_config(value)


def test_training_rejects_parameter_mismatch_before_and_after_cuda_transfer(monkeypatch):
    class Model:
        def parameters(self):
            return [type("Parameter", (), {"device": type("Device", (), {"type": "cpu"})()})()]

    model = Model()
    transfers = []
    monkeypatch.setattr(training, "_transfer_initialized_model_to_cuda", transfers.append)
    monkeypatch.setattr(training, "_SOURCE_PARAMETER_SHA256", lambda _model: "0" * 64)
    with pytest.raises(RuntimeError, match="before training"):
        training._verified_initial_parameter_sha256(model)
    assert transfers == []

    hashes = iter((run_config.EXPECTED_INITIAL_PARAMETER_SHA256, "0" * 64))
    monkeypatch.setattr(training, "_SOURCE_PARAMETER_SHA256", lambda _model: next(hashes))
    with pytest.raises(RuntimeError, match="after the CPU-to-CUDA transfer"):
        training._verified_initial_parameter_sha256(model)
    assert transfers == [model]


@pytest.mark.parametrize(
    ("worker", "expected_peak", "expected_minimum"),
    (("a0-lr-6e-4", 6e-4, 6e-5), ("a0-lr-1e-3", 1e-3, 1e-4)),
)
def test_real_run_worker_dispatch_forwards_concrete_condition_lr(
    worker: str,
    expected_peak: float,
    expected_minimum: float,
    monkeypatch,
    tmp_path: Path,
):
    captured = {}

    def fake_condition(**kwargs):
        captured.update(kwargs)
        return tmp_path / f"attempt-{worker}"

    monkeypatch.setattr(training, "_ORIGINAL_RUN_CONDITION", fake_condition)
    monkeypatch.setattr(training._BASE, "require_cuda", lambda _torch: None)
    # The dispatch contract is independent of the immutable attempt-slot guard.
    # Isolate it so this regression remains runnable after terminal artifacts exist.
    monkeypatch.setattr(
        training._BASE, "_require_attempt_slots_available", lambda _conditions: None
    )
    monkeypatch.setattr(
        training._BASE,
        "load_verified_caches",
        lambda _config, *, np: (object(), object(), {}, {}, 0.0),
    )
    monkeypatch.setattr(
        training._BASE,
        "build_schedule",
        lambda _config, _metadata, *, np: (object(), "schedule", {}),
    )
    monkeypatch.setattr(training._BASE, "run_code_identity", lambda: {})
    monkeypatch.setattr(training._BASE, "git_identity", lambda: {})
    monkeypatch.setattr(training._BASE, "write_json", lambda *_args, **_kwargs: None)

    source = run_config.load_config()
    assert source["training"]["peak_learning_rate"] is None
    results = training.run_worker(worker)

    assert results == [tmp_path / f"attempt-{worker}"]
    assert training._BASE.run_condition is training._run_condition_with_resolved_training
    assert captured["condition"]["id"] == worker
    assert captured["config"]["training"]["peak_learning_rate"] == expected_peak
    assert captured["config"]["training"]["minimum_learning_rate"] == pytest.approx(
        expected_minimum
    )
    assert source["training"]["peak_learning_rate"] is None


def test_training_row_uses_exact_final_64_boundaries(tmp_path: Path):
    attempt = tmp_path / "attempt"
    attempt.mkdir()
    rows = []
    for step in range(1, 713):
        rows.append(
            {
                "event": "train",
                "step": step,
                "task_loss": 10.0 if step < 649 else float(step),
                "learning_rate": 1e-4,
                "adamw_gradient_norm_pre_clip": 2.0,
                "adamw_gradient_norm_post_clip": 1.0,
                "adamw_gradient_was_clipped": True,
                "gradient_overflow": False,
                "optimizer_step_skipped": False,
            }
        )
    (attempt / "events.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    result = selection._training_row(
        attempt,
        condition_id="arm",
        peak=6e-4,
        minimum=6e-5,
        validation_loss=-999.0,
        r_model=0.0,
        provenance="test",
    )
    assert result["stable"] is True
    assert result["selection_metric"] == pytest.approx(sum(range(649, 713)) / 64)
    assert result["gradient_clipped_boundary_count"] == 712
    assert result["validation_loss_not_used_for_selection"] == -999.0


def _arm(condition_id: str, peak: float, metric: float, *, stable: bool = True) -> dict:
    return {
        "condition_id": condition_id,
        "peak_learning_rate": peak,
        "selection_metric": metric,
        "stable": stable,
        "provenance": "run019_pinned_baseline" if peak == 3e-4 else "run021_new_full_pass",
        "validation_loss_not_used_for_selection": -1e9,
    }


def test_selection_rule_selects_bracketed_arm_and_ignores_unstable_arm():
    rows = [_arm("a0-gelu", 3e-4, 4.5), _arm("a0-lr-6e-4", 6e-4, 4.2)]
    rows.append(_arm("a0-lr-1e-3", 1e-3, 1.0, stable=False))
    status, selected, best, action = selection.selection_decision(rows)
    assert (status, selected, best["condition_id"]) == (
        "selected",
        "a0-lr-6e-4",
        "a0-lr-6e-4",
    )
    assert "TEAL" in action


def test_selection_rule_defers_when_one_e_minus_three_is_unbracketed_best():
    rows = [_arm("a0-gelu", 3e-4, 4.5), _arm("a0-lr-6e-4", 6e-4, 4.2)]
    rows.append(_arm("a0-lr-1e-3", 1e-3, 4.0))
    status, selected, best, action = selection.selection_decision(rows)
    assert status == "deferred_unbracketed"
    assert selected is None
    assert best["condition_id"] == "a0-lr-1e-3"
    assert "1.5e-3" in action


def test_teal_gate_accepts_only_selected_new_arm(tmp_path: Path, monkeypatch):
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    (artifacts / "selection.json").write_text(
        json.dumps({"status": "selected", "selected_condition_id": "a0-lr-6e-4"}),
        encoding="utf-8",
    )
    config = run_config.load_config()
    monkeypatch.setattr(teal_posthoc, "RUN_DIR", tmp_path)
    monkeypatch.setattr(teal_posthoc, "load_config", lambda: config)
    assert teal_posthoc.selected_new_condition() == "a0-lr-6e-4"
    (artifacts / "selection.json").write_text(
        json.dumps({"status": "selected", "selected_condition_id": "a0-gelu"}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Run 019 baseline"):
        teal_posthoc.selected_new_condition()


def test_r_model_fraction_is_not_bounded_by_a0_reach_ceiling():
    verification._require_r_model_fraction(0.5, "attempt")
    with pytest.raises(ValueError, match=r"outside \[0,1\]"):
        verification._require_r_model_fraction(1.01, "attempt")


def test_run_code_identity_is_direct_complete_and_lf_canonicalized():
    identity = run_config.run_code_identity()
    paths = {row["path"] for row in identity["files"]}
    assert "selection.py" in paths
    assert "09_verify_worker_entrypoint.py" in paths
    assert "launch-control/prelaunch/verify_worker.sh" in paths
    assert "../004-2026-08-29-pythia14m-full-pass-l1n/training.py" in paths
    assert identity["newline_policy"].startswith("text inputs canonicalized")
    assert len(identity["content_sha256"]) == 64


def test_expected_a0_ceiling_is_exact():
    ceiling = run_config.expected_ceiling("A0")
    assert ceiling["reachable_product_count"] == 0
    assert ceiling["model_product_count"] == 827_099_971_584
    assert math.isclose(ceiling["R_model_max_fraction"], 0.0)

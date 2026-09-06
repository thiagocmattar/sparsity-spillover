"""CPU-only probe-contract checks; no CUDA call or compiler invocation."""
import importlib.util
import json
from pathlib import Path

import pytest
import torch


spec = importlib.util.spec_from_file_location("k001_probe_test_target", Path(__file__).with_name("gpu_probe.py"))
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def test_original_48_and_amplitude_stress_case_identity():
    expected_shapes = sorted({(3, k, n) for d in (128, 512, 1024) for k, n in
                              ((d, 3 * d), (d, 4 * d), (4 * d, d), (d, d))}
                             | {(3, 33, 129), (3, 32, 2048), (3, 64, 2048), (256, 512, 128)})
    cases = probe.primitive_cases()
    assert len(cases) == 80
    assert [(row["m"], row["k"], row["n"], row["pattern"]) for row in cases[:48]] == [
        (*shape, pattern) for shape in expected_shapes for pattern in ("zero", "signed_sparse", "full")]
    assert all(row["phase"] == "original_48" and row["activation_scale"] == .1 for row in cases[:48])
    assert all(row["phase"] == "amplitude_stress" and row["pattern"] == "signed_sparse" for row in cases[48:])
    assert {row["activation_scale"] for row in cases} == {.1, 1., 10.}


def test_timing_matrix_covers_sizes_sites_and_zero_fractions():
    cases = probe.timing_cases()
    assert len(cases) == 48
    assert {row["m"] for row in cases} == {2048}
    assert {row["model_size"] for row in cases} == {"14m", "70m", "410m"}
    assert {row["site"] for row in cases} == {"a", "m", "h", "z"}
    assert {row["requested_zero_fraction"] for row in cases} == {0., .5, .9, .99}


def test_fixed_imported_gate_rejects_large_error():
    cfg = json.loads((probe.RUN / "config.json").read_text())["calibration"]
    reference = torch.ones(2, 3)
    assert probe.check_output(reference, reference.clone(), cfg)["pass"]
    assert not probe.check_output(reference, reference * 2, cfg)["pass"]


def test_recorder_immutable_attempt_and_failure_preservation(tmp_path):
    recorder = probe.Recorder(tmp_path / "attempt", 10)
    recorder.emit("failed", error="synthetic CPU fixture", complete=False)
    assert json.loads((recorder.directory / "status.json").read_text())["stage"] == "failed"
    assert json.loads((recorder.directory / "events.jsonl").read_text())["complete"] is False
    with pytest.raises(FileExistsError):
        probe.Recorder(recorder.directory, 10)

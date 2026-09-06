import importlib.util
from pathlib import Path


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("run025_k005_probe", HERE / "gpu_probe.py")
P = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(P)


def test_probe_covers_width_specializations_and_amplitude_stress():
    cases = P.primitive_cases()
    assert any(row["n"] == 128 for row in cases)
    assert any(row["n"] >= 512 for row in cases)
    assert any(row["activation_scale"] == 10.0 for row in cases)


def test_timing_directly_compares_all_family_shapes():
    cases = P.timing_cases()
    assert len(cases) == 48
    assert {row["model_size"] for row in cases} == {"14m", "70m", "410m"}
    assert {row["site"] for row in cases} == {"a", "m", "h", "z"}
    assert {row["requested_zero_fraction"] for row in cases} == {0.0, 0.8, 0.95, 0.99}

import importlib.util
from pathlib import Path


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("run025_k004_probe", HERE / "gpu_probe.py")
P = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(P)


def test_primitive_matrix_covers_both_strategies_and_signed_stress():
    cases = P.primitive_cases()
    assert {row["strategy"] for row in cases} == {"inline", "flagged"}
    assert any(row["scale"] == 10.0 for row in cases)
    assert any(row["k"] == 2048 for row in cases)


def test_timing_matrix_covers_all_14m_projection_shapes_and_tile_levels():
    cases = P.timing_cases()
    assert len(cases) == 16
    assert {row["site"] for row in cases} == {"a", "m", "h", "z"}
    assert {row["tile_zero_fraction"] for row in cases} == {0.0, 0.5, 0.8, 0.95}

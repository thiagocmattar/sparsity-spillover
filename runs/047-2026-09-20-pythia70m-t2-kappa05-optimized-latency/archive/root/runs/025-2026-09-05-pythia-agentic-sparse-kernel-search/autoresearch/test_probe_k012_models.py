import ast
from pathlib import Path


HERE = Path(__file__).resolve().parent


def test_k012_probe_is_single_candidate_and_records_parent_kernel():
    source = (HERE / "probe_k012_models.py").read_text(encoding="utf-8")
    ast.parse(source)
    assert 'base.IMPLEMENTATIONS = ("k012",)' in source
    assert 'HERE / "candidates/k012/candidate.py"' in source
    assert 'HERE / "candidates/k001/kernel.cu"' in source

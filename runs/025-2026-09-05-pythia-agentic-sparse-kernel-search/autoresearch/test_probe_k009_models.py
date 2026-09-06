import ast
from pathlib import Path


HERE = Path(__file__).resolve().parent


def test_k009_probe_records_frozen_policy_and_inherited_kernel():
    source = (HERE / "probe_k009_models.py").read_text()
    ast.parse(source)
    assert 'base.IMPLEMENTATIONS = ("k009",)' in source
    assert 'HERE / "candidates/k009/candidate.py"' in source
    assert 'HERE / "candidates/k001/kernel.cu"' in source

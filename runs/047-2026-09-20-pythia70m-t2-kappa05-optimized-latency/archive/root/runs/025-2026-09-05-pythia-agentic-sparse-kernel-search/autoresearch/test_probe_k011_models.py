import ast
from pathlib import Path


HERE = Path(__file__).resolve().parent


def test_k011_probe_requires_named_static_mask_and_records_parent_kernel():
    source = (HERE / "probe_k011_models.py").read_text(encoding="utf-8")
    ast.parse(source)
    assert 'MASK_ENV = "RUN025_K011_MASK"' in source
    assert "module.selected_layers(identifier)" in source
    assert 'HERE / "candidates/k011/candidate.py"' in source
    assert 'HERE / "candidates/k001/kernel.cu"' in source

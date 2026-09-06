import ast
from pathlib import Path


HERE = Path(__file__).resolve().parent


def test_k008_probe_requires_named_static_mask_and_records_inherited_kernel():
    source = (HERE / "probe_k008_models.py").read_text()
    ast.parse(source)
    assert 'MASK_ENV = "RUN025_K008_MASK"' in source
    assert "module.selected_layers(identifier)" in source
    assert 'HERE / "candidates/k008/candidate.py"' in source
    assert 'HERE / "candidates/k001/kernel.cu"' in source

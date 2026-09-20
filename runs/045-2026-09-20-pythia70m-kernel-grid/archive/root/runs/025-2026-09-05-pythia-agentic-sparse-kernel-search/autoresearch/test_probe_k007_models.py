import ast
from pathlib import Path


HERE = Path(__file__).resolve().parent


def test_k007_probe_requires_named_environment_config_and_records_sources():
    source = (HERE / "probe_k007_models.py").read_text()
    ast.parse(source)
    assert 'CONFIG_ENV = "RUN025_K007_CONFIG"' in source
    assert "module.geometry(identifier)" in source
    assert 'HERE / "candidates/k007/candidate.py"' in source
    assert 'HERE / "candidates/k007/kernels.py"' in source

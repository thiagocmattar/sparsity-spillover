import ast
from pathlib import Path


HERE = Path(__file__).resolve().parent


def test_k001_z_probe_is_fixed_and_records_its_entrypoint():
    source = (HERE / "probe_k001_z_models.py").read_text()
    ast.parse(source)
    assert 'FIXED_SITES = frozenset(("z",))' in source
    assert 'base.IMPLEMENTATIONS = ("k001",)' in source
    assert "base.record(Path(__file__))" in source
    assert "base.select_sites = select_sites" in source

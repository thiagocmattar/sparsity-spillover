import ast
from pathlib import Path


HERE = Path(__file__).resolve().parent


def test_k004_z_probe_is_fixed_to_inline_z_and_records_sources():
    source = (HERE / "probe_k004_z_models.py").read_text()
    tree = ast.parse(source)
    names = {
        node.targets[0].id: ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "STRATEGY"
    }
    assert names["STRATEGY"] == "inline"
    assert 'FIXED_SITES = frozenset(("z",))' in source
    assert "base.select_sites = select_sites" in source
    assert 'HERE / "candidates/k004/kernels.py"' in source

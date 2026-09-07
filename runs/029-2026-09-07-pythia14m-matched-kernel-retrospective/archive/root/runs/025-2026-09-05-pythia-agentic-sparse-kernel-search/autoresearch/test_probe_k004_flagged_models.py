import ast
from pathlib import Path


HERE = Path(__file__).resolve().parent


def test_k004_flagged_probe_is_fixed_and_records_both_kernel_sources():
    source = (HERE / "probe_k004_flagged_models.py").read_text()
    tree = ast.parse(source)
    assignments = {
        node.targets[0].id: ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "STRATEGY"
    }
    assert assignments == {"STRATEGY": "flagged"}
    assert 'HERE / "probe_models.py"' in source
    assert 'HERE / "candidates/k004/candidate.py"' in source
    assert 'HERE / "candidates/k004/kernels.py"' in source

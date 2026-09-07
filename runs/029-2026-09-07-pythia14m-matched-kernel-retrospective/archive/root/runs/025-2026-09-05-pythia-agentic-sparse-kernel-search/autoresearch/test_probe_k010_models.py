import ast
from pathlib import Path


SOURCE = Path(__file__).with_name("probe_k010_models.py")


def test_k010_probe_records_policy_and_inherited_kernel_sources():
    text = SOURCE.read_text(encoding="utf-8")
    ast.parse(text)
    assert 'base.IMPLEMENTATIONS = ("k010",)' in text
    assert 'HERE / "candidates/k010/candidate.py"' in text
    assert 'HERE / "candidates/k004/kernels.py"' in text
    assert 'FIXED_SITES = frozenset(("z",))' in text

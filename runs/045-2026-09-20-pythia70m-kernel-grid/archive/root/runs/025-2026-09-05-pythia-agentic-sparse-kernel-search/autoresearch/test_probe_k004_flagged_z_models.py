import ast
from pathlib import Path


SOURCE = Path(__file__).with_name("probe_k004_flagged_z_models.py")


def test_flagged_z_probe_has_fixed_site_and_strategy():
    text = SOURCE.read_text(encoding="utf-8")
    ast.parse(text)
    assert 'STRATEGY = "flagged"' in text
    assert 'FIXED_SITES = frozenset(("z",))' in text
    assert 'base.IMPLEMENTATIONS = ("k004",)' in text

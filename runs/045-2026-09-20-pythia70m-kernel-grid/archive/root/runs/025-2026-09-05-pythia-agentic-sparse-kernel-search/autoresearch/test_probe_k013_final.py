from pathlib import Path


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "probe_k013_final.py"


def test_k013_final_specializes_immutable_evaluator_and_freeze():
    text = SOURCE.read_text(encoding="utf-8")
    assert '"size": "14m"' in text
    assert '"freeze": "candidates/k013/FROZEN.json"' in text
    assert '"site_policy": "topology"' in text
    assert 'base.IMPLEMENTATIONS = ("k013",)' in text
    assert "base.main()" in text

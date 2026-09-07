from pathlib import Path


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "probe_k016_final.py"


def test_k016_final_uses_frozen_complete_validation_evaluator():
    text = SOURCE.read_text(encoding="utf-8")
    base = (HERE / "probe_frozen_final.py").read_text(encoding="utf-8")
    assert '"k016"' in text and '"size": "70m"' in text
    assert '"site_policy": "topology"' in text
    assert "if not args.full_validation" in base
    assert "blocks != 338 or tail != 1444" in base

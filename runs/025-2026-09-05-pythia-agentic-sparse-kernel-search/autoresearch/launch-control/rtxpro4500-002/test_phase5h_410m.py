from pathlib import Path
import re


SCRIPT = Path(__file__).with_name("phase5h-410m-k010-heldout-full-validation.sh")


def test_phase5h_is_exact_six_interior_kappas_with_full_validation():
    source = SCRIPT.read_text(encoding="utf-8")
    conditions = re.findall(r"run_probe (410m/\S+) ", source)
    assert conditions == [
        "410m/a4-0p01", "410m/a4-0p05", "410m/a4-0p1",
        "410m/a7-0p01", "410m/a7-0p05", "410m/a7-0p1",
    ]
    assert "--full-validation" in source
    assert "--inputs 16 --passes 5" in source
    assert "probe_k010_heldout.py" in source

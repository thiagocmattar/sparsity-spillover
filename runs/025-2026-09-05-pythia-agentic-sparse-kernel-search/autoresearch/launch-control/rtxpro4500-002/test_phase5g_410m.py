from pathlib import Path


SCRIPT = Path(__file__).with_name("phase5g-410m-k010-other-development.sh")


def test_phase5g_uses_only_the_three_unused_development_endpoints():
    source = SCRIPT.read_text(encoding="utf-8")
    assert source.count("run_probe 410m/") == 3
    assert "run_probe 410m/a1h a1h" in source
    assert "run_probe 410m/a4-0 a4-0" in source
    assert "run_probe 410m/a7-0 a7-0" in source
    assert "0p5" not in source
    assert "--inputs 16 --passes 5" in source

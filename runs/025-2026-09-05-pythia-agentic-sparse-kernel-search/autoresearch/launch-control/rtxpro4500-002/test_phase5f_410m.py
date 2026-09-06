from pathlib import Path


SCRIPT = Path(__file__).with_name("phase5f-410m-k010-three-development.sh")


def test_phase5f_is_one_frozen_policy_on_three_development_endpoints():
    source = SCRIPT.read_text(encoding="utf-8")
    assert source.count("run_probe 410m/") == 3
    assert "run_probe 410m/a0 a0" in source
    assert "run_probe 410m/a4-0p5 a4-0p5" in source
    assert "run_probe 410m/a7-0p5 a7-0p5" in source
    assert "probe_k010_models.py" in source
    assert "--inputs 16 --passes 5" in source

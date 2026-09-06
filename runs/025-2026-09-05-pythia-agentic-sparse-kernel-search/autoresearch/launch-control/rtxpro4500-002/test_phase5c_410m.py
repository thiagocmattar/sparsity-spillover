from pathlib import Path


SCRIPT = Path(__file__).with_name("phase5c-410m-high-site-isolation.sh")


def test_phase5c_is_fixed_to_h_and_z_at_two_high_endpoints():
    source = SCRIPT.read_text(encoding="utf-8")
    assert source.count("run_site h 410m/") == 2
    assert source.count("run_site z 410m/") == 2
    assert "410m/a4-0p5" in source and "410m/a7-0p5" in source
    assert "410m/a4-0 " not in source and "410m/a7-0 " not in source
    assert "--inputs 16 --passes 5" in source
    assert "probe_k004_z_models.py" in source

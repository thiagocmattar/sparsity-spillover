from pathlib import Path


SCRIPT = Path(__file__).with_name("phase5e-410m-flagged-h.sh")


def test_phase5e_is_two_high_endpoint_flagged_h_probes():
    source = SCRIPT.read_text(encoding="utf-8")
    assert source.count("run_probe 410m/") == 2
    assert "410m/a4-0p5" in source and "410m/a7-0p5" in source
    assert "probe_k004_flagged_models.py" in source
    assert "--sites h" in source
    assert "--inputs 16 --passes 5" in source

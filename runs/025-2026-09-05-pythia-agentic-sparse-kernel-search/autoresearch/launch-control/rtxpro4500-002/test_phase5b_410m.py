from pathlib import Path


SCRIPT = Path(__file__).with_name("phase5b-410m-high-occupancy.sh")


def test_phase5b_is_only_two_high_pressure_development_endpoints():
    source = SCRIPT.read_text(encoding="utf-8")
    assert source.count("run_occupancy 410m/") == 2
    assert "run_occupancy 410m/a4-0p5 a4-0p5" in source
    assert "run_occupancy 410m/a7-0p5 a7-0p5" in source
    assert "410m/a4-0 " not in source
    assert "410m/a7-0 " not in source
    assert "--seconds 180" in source

from pathlib import Path


SCRIPT = Path(__file__).with_name("phase6c-14m-k011-finalist-confirmation.sh")


def test_phase6c_confirms_only_screening_tied_finalists():
    source = SCRIPT.read_text(encoding="utf-8")
    assert "for run025_mask in suffix4 suffix5; do" in source
    assert source.count("run_probe 14m/") == 3
    assert "--inputs 16 --passes 10" in source
    assert "0p01" not in source and "0p05" not in source

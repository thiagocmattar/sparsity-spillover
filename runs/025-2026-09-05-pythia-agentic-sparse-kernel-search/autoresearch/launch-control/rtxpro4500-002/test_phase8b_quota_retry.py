from pathlib import Path


HERE = Path(__file__).resolve().parent
SCRIPT = HERE / "phase8b-14m-k011-quota-retry.sh"


def test_retry_uses_new_attempts_and_never_heldout_conditions():
    text = SCRIPT.read_text(encoding="utf-8")
    assert "k011retry-14m-" in text
    assert "--full-validation" in text
    assert "14m/a4-0p01" not in text
    assert "14m/a7-0p01" not in text
    assert "for run025_mask in suffix4 odd" in text


def test_retry_verifies_and_reuses_completed_suffix3_screen_artifacts():
    text = SCRIPT.read_text(encoding="utf-8")
    assert "require_complete" in text
    assert "k011full-14m-a0-suffix3-rtxpro4500-002" in text
    assert "k011full-14m-a4-0-suffix3-rtxpro4500-002" in text
    assert "run_probe 14m/a7-0 active a7-0 suffix3" in text

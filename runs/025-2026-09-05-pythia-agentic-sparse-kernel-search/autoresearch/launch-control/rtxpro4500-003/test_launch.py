from pathlib import Path


SCRIPT = Path(__file__).with_name("launch.ps1").read_text()


def test_launch_reuses_volume_and_has_a_hard_guard() -> None:
    assert "--network-volume-id '9luykg5yc3'" in SCRIPT
    assert "--data-center-ids 'EUR-IS-1'" in SCRIPT
    assert ".AddHours(3)" in SCRIPT
    assert "06_billing_guard.ps1" in SCRIPT
    assert "-WindowStyle Hidden" in SCRIPT


def test_launch_preserves_budget_reserve() -> None:
    assert "maximum_gpu_cost_usd = 2.16" in SCRIPT
    assert "protected_h100_and_teardown_reserve_usd = 10" in SCRIPT
    assert "study_total_usd_max = 40" in SCRIPT


def test_launch_is_idempotence_guarded() -> None:
    assert "Lease already recorded" in SCRIPT
    assert "discover the exact name before retrying" in SCRIPT

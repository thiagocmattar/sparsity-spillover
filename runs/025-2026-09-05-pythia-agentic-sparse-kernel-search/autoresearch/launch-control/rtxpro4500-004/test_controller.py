from pathlib import Path


HERE = Path(__file__).parent
LAUNCH = (HERE / "launch.ps1").read_text()
TRANSFER = (HERE / "transfer-and-start.ps1").read_text()
BOOTSTRAP = (HERE / "bootstrap-and-run.sh").read_text()
CLOSEOUT = (HERE / "closeout.ps1").read_text()


def test_launch_has_exact_gpu_volume_and_two_hour_guard() -> None:
    assert "NVIDIA RTX PRO 4500 Blackwell" in LAUNCH
    assert "9luykg5yc3" in LAUNCH
    assert "AddHours(2)" in LAUNCH
    assert "maximum_gpu_cost_usd = 1.44" in LAUNCH
    assert "Start-Process powershell.exe" in LAUNCH
    assert "-WindowStyle Hidden" in LAUNCH


def test_transfer_is_small_direct_scp_and_identity_checked() -> None:
    assert "run025-phase17-code.tar.gz" not in TRANSFER  # resolved through frozen manifest
    assert "Get-FileHash" in TRANSFER
    assert "scp.exe" in TRANSFER
    assert "no model or token-cache transfer" in TRANSFER
    assert "StrictHostKeyChecking=yes" in TRANSFER


def test_bootstrap_reuses_runtime_and_requires_blackwell() -> None:
    assert "artifacts/runtime/venv-pythia/bin/python" in BOOTSTRAP
    assert "major == 12" in BOOTSTRAP
    assert "6300s" in BOOTSTRAP
    assert "phase17-fixed-rmodel-pairs.sh" in BOOTSTRAP


def test_closeout_verifies_before_deleting_exact_pod() -> None:
    assert CLOSEOUT.index("verify_phase17_archive.py") < CLOSEOUT.index("pod delete")
    assert "Local archive hash does not match the remote hash" in CLOSEOUT
    assert "pod get $run025Lease.pod_id" in CLOSEOUT
    assert "37 Phase 17 artifact directories" in CLOSEOUT

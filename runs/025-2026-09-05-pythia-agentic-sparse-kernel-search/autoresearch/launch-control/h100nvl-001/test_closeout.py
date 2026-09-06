from pathlib import Path


HERE = Path(__file__).resolve().parent


def test_verifier_enforces_inventory_phase_and_directory_count():
    text = (HERE / "verify_h100_evidence.py").read_text(encoding="utf-8")
    assert 'extractall(args.destination, filter="data")' in text
    assert "evidence-inventory.json" in text
    assert "digest(path) != row" in text
    assert "len(directories) != 93" in text
    assert "all-checks-finished-utc.txt" in text
    assert "Phase 16 did not exit successfully" in text


def test_closeout_orders_remote_and_local_verification_before_deletion():
    text = (HERE / "closeout.ps1").read_text(encoding="utf-8")
    remote_ready = text.index("evidence-ready-utc.txt")
    remote_hash = text.index("sha256sum")
    transfer = text.index("scp.exe")
    local_hash = text.index("Get-FileHash")
    inventory_verify = text.index("verify_h100_evidence.py")
    deletion = text.index("pod delete")
    absent = text.index("pod get", deletion)
    assert remote_ready < remote_hash < transfer < local_hash < inventory_verify < deletion < absent
    assert "Pod remains running" in text


def test_closeout_is_scoped_and_does_not_delete_persistent_storage():
    text = (HERE / "closeout.ps1").read_text(encoding="utf-8")
    assert "$run025Target.name -ne $run025Lease.name" in text
    assert "pod delete $run025Lease.pod_id" in text
    assert "network volume" not in text.lower()

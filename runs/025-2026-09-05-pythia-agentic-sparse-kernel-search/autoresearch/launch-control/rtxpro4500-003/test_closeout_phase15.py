from pathlib import Path


HERE = Path(__file__).resolve().parent


def test_closeout_orders_verification_before_scoped_deletion():
    text = (HERE / "closeout-phase15.ps1").read_text(encoding="utf-8")
    remote_complete = text.index("all-checks-finished-utc.txt")
    remote_package = text.index("collect_evidence.py")
    local_verify = text.index("--extract-verify")
    deletion = text.index("pod delete")
    not_found = text.index("pod get", deletion)
    assert remote_complete < remote_package < local_verify < deletion < not_found
    assert "$run025Lease.pod_id" in text
    assert "$run025Target.name -ne $run025Lease.name" in text


def test_closeout_does_not_delete_storage_or_continue_after_verification_failure():
    text = (HERE / "closeout-phase15.ps1").read_text(encoding="utf-8")
    assert "network volume" not in text.lower()
    assert "pod delete $run025Lease.pod_id" in text
    assert "Local evidence verification failed; Pod remains running" in text
    assert "Retrieval destination already exists" in text

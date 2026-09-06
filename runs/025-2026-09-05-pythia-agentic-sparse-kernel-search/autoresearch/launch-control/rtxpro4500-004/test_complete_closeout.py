from pathlib import Path


HERE = Path(__file__).parent
CLOSEOUT = (HERE / "closeout-complete.ps1").read_text()
VERIFY18 = (HERE / "verify_phase18_archive.py").read_text()


def test_complete_closeout_accounts_both_phases() -> None:
    assert "phase17-fixed-rmodel-pairs-rtxpro4500-004" in CLOSEOUT
    assert "phase18-14m-k001-confirmation-rtxpro4500-004" in CLOSEOUT
    assert "Expected 50 Phase 17/18 artifact directories" in CLOSEOUT
    assert "verify_phase17_archive.py" in CLOSEOUT
    assert "verify_phase18_archive.py" in CLOSEOUT


def test_complete_closeout_verifies_before_deletion() -> None:
    assert CLOSEOUT.index("verify_phase17_archive.py") < CLOSEOUT.index("pod delete")
    assert CLOSEOUT.index("verify_phase18_archive.py") < CLOSEOUT.index("pod delete")
    assert "Local archive hash does not match the remote hash" in CLOSEOUT


def test_phase18_verifier_requires_full_coverage_and_fixed_identity() -> None:
    assert "692224" in VERIFY18
    assert "1444" in VERIFY18
    assert "Expected 80 paired timing samples" in VERIFY18
    assert "Checkpoint or R_model changed" in VERIFY18
    assert 'for repeat in (1, 2, 3)' in VERIFY18

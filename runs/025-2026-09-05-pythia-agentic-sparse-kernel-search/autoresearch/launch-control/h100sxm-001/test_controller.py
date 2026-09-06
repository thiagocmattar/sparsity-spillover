from pathlib import Path


HERE = Path(__file__).resolve().parent


def test_bootstrap_verifies_before_extracting_and_overlays_code_last():
    text = (HERE / "bootstrap-and-run.sh").read_text(encoding="utf-8")
    verify = text.index("sha256sum -c SHA256SUMS")
    old_code = text.index("tar -xzf payload.tar.gz")
    current_code = text.index("tar -xzf run025-h100-code.tar.gz")
    runtime = text.index("uv pip install")
    phase = text.index("phase16-h100-frozen-transfer.sh")
    assert verify < old_code < current_code < runtime < phase
    assert "torch.cuda.get_device_capability()[0] == 9" in text
    assert "9000s" in text
    assert '"${#run025_artifacts[@]}" -ne 93' in text


def test_launch_is_bounded_and_has_no_persistent_volume():
    text = (HERE / "launch.ps1").read_text(encoding="utf-8")
    assert "NVIDIA H100 80GB HBM3" in text
    assert "--cloud-type COMMUNITY" in text
    assert "--public-ip" in text
    assert ".AddHours(3)" in text
    assert "--volume-in-gb 60" in text
    assert "--network-volume-id" not in text
    assert "06_billing_guard.ps1" in text


def test_code_bundle_is_commit_derived_and_inputs_are_pinned():
    text = (HERE / "prepare-code.ps1").read_text(encoding="utf-8")
    assert "git -C $run025Root archive" in text
    assert "rev-parse HEAD" in text
    for digest in (
        "a0867d2211b0a769b990a19fa78b154be4b087842d2c3cbf37304d7dc8767a39",
        "fa74f92130fd511f4f98206cf2ca9d2446afa7fe8ca9b20ea5d21a3f394deb6c",
        "4d7168518eb307e7e05d963f6efbac7ab4091ba7db17d56bbbfcb02a8bae79fd",
    ):
        assert digest in text


def test_transfer_never_records_ephemeral_code():
    text = (HERE / "transfer-and-start.ps1").read_text(encoding="utf-8")
    assert "RunPod encrypted one-time-code relay" in text
    assert "sha256sum -c SHA256SUMS" in text
    result_section = text.lower().split("$run025start =", 1)[1]
    assert "actualcode" not in result_section
    assert "start.json" in text

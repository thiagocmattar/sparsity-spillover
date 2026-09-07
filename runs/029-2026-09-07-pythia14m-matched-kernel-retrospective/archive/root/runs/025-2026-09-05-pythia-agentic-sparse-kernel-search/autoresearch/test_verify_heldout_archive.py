import importlib.util
from pathlib import Path


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "run025_verify_heldout", HERE / "verify_heldout_archive.py"
)
V = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(V)


def test_safe_member_name_rejects_absolute_and_parent_paths():
    assert V.safe_member_name("runs/025/file")
    assert not V.safe_member_name("/absolute")
    assert not V.safe_member_name("../escape")

import hashlib
import importlib.util
import io
from pathlib import Path

import pytest


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "run025_extract_70m", HERE / "extract_70m_development.py"
)
E = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(E)


def test_write_member_is_exclusive_and_accepts_only_exact_existing(tmp_path):
    payload = b"retained checkpoint"
    expected = {
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    target = tmp_path / "nested/model.safetensors"
    assert E.write_member(io.BytesIO(payload), target, expected) == "extracted"
    assert E.write_member(io.BytesIO(b"unused"), target, expected) == "already_exact"
    target.write_bytes(b"different")
    with pytest.raises(FileExistsError, match="another identity"):
        E.write_member(io.BytesIO(payload), target, expected)


def test_safe_target_rejects_absolute_and_parent_paths():
    with pytest.raises(ValueError, match="Unsafe"):
        E.safe_target("/absolute")
    with pytest.raises(ValueError, match="Unsafe"):
        E.safe_target("../escape")

import json

import pytest

from sparsity_research.artifacts import (
    attempt_lifecycle,
    build_transfer_inventory,
    start_attempt,
    verify_transfer_inventory,
)


def test_completed_attempt_publishes_terminal_manifest_last(tmp_path):
    with attempt_lifecycle(
        tmp_path,
        config={"model": {"initialization": "random"}},
        command="python 02_train.py",
        mode="pretrain",
    ) as attempt:
        attempt.append_event({"event": "train", "step": 1})
        attempt.complete(metrics={"loss": 1.0}, predictions=[])
    manifest = json.loads((attempt.attempt_dir / "manifest.json").read_text())
    assert manifest["status"] == "completed"
    assert json.loads((attempt.attempt_dir / "metrics.json").read_text()) == {"loss": 1.0}
    assert (attempt.attempt_dir / "events.jsonl").read_text().strip()


def test_escaping_error_is_preserved_as_failed_attempt(tmp_path):
    with pytest.raises(RuntimeError, match="boom"):
        with attempt_lifecycle(
            tmp_path,
            config={"x": 1},
            command="test",
            mode="diagnostic",
        ):
            raise RuntimeError("boom")
    attempt_dir = next((tmp_path / "artifacts" / "attempts").iterdir())
    manifest = json.loads((attempt_dir / "manifest.json").read_text())
    assert manifest["status"] == "failed"
    assert manifest["failure"]["message"] == "boom"


def test_lifecycle_requires_explicit_terminal_state(tmp_path):
    with pytest.raises(RuntimeError, match="without"):
        with attempt_lifecycle(tmp_path, config={"x": 1}, command="test", mode="test"):
            pass


def test_transfer_inventory_detects_changed_bytes(tmp_path):
    (tmp_path / "result.json").write_text('{"ok": true}\n', encoding="utf-8")
    inventory = build_transfer_inventory(tmp_path)
    verify_transfer_inventory(tmp_path, inventory)
    (tmp_path / "result.json").write_text('{"ok": false}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="mismatch"):
        verify_transfer_inventory(tmp_path, inventory)


def test_attempt_ids_are_monotonic(tmp_path):
    first = start_attempt(tmp_path, config={"x": 1}, command="x", mode="test")
    first.fail(RuntimeError("first"))
    second = start_attempt(tmp_path, config={"x": 1}, command="x", mode="test")
    second.fail(RuntimeError("second"))
    assert first.attempt_id.startswith("001-")
    assert second.attempt_id.startswith("002-")


def test_explicit_attempt_sequence_supports_disjoint_worker_merge(tmp_path):
    attempt = start_attempt(
        tmp_path,
        config={"condition": "worker-five"},
        command="test",
        mode="pretrain",
        attempt_sequence=5,
    )
    attempt.fail(RuntimeError("deliberate terminal state"))
    assert attempt.attempt_id.startswith("005-")
    with pytest.raises(FileExistsError, match="005"):
        start_attempt(
            tmp_path,
            config={"condition": "collision"},
            command="test",
            mode="pretrain",
            attempt_sequence=5,
        )
    with pytest.raises(ValueError, match=r"\[1, 999\]"):
        start_attempt(
            tmp_path,
            config={"condition": "invalid"},
            command="test",
            mode="pretrain",
            attempt_sequence=0,
        )

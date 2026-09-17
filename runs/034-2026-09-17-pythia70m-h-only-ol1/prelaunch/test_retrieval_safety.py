"""Destructive infrastructure actions must remain behind artifact verification."""
from contextlib import nullcontext
import importlib.util
import json
import hashlib
from pathlib import Path
from types import SimpleNamespace
import tarfile

import pytest

spec = importlib.util.spec_from_file_location('retrieval034', Path(__file__).with_name('monitor_and_retrieve.py'))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
sync_function = module.sync_ready_checkpoints


@pytest.fixture
def setup(tmp_path, monkeypatch):
    records = tmp_path / 'prelaunch'
    records.mkdir()
    monkeypatch.setattr(module, 'HERE', tmp_path)
    monkeypatch.setattr(module, 'RECORDS', records)
    monkeypatch.setattr(module, 'sync_ready_checkpoints', lambda *args, **kwargs: None)
    module.cloud.write(records / 'assignments.json', {'sample': ['a4-h-ol1-kappa-0']})
    module.cloud.write(records / 'lease-sample.json', {'pod': {'id': 'owned123'}})
    calls = []
    monkeypatch.setattr(module.cloud, 'cli', lambda *args: calls.append(args))
    return records, calls


def saved_receipt(records, **extra):
    target = records / 'retrieval/sample/receipt.json'
    module.cloud.write(target, {'locally_verified': True, 'pod_id': 'owned123', **extra})
    return target


def test_incomplete_remote_condition_prevents_deletion(setup, monkeypatch):
    _, calls = setup
    monkeypatch.setattr(module, 'remote_status', lambda label: {'conditions': [{'verified': False, 'exit_code': '0'}]})
    with pytest.raises(AssertionError):
        module.retrieve_and_delete('sample')
    assert calls == []


def test_wrong_receipt_identity_prevents_api_call(setup):
    records, calls = setup
    saved_receipt(records, pod_id='different')
    with pytest.raises(AssertionError):
        module.retrieve_and_delete('sample')
    assert calls == []


def test_wrong_live_pod_name_prevents_deletion(setup, monkeypatch):
    records, calls = setup
    saved_receipt(records)
    def cli(*args):
        calls.append(args)
        return [{'id': 'owned123', 'name': 'unrelated'}]
    monkeypatch.setattr(module.cloud, 'cli', cli)
    with pytest.raises(AssertionError):
        module.retrieve_and_delete('sample')
    assert calls == [('pod', 'list', '--all')]


def test_verified_owned_pod_deleted_then_absence_confirmed(setup, monkeypatch):
    records, calls = setup
    target = saved_receipt(records)
    def cli(*args):
        calls.append(args)
        if len(calls) == 1:
            return [{'id': 'owned123', 'name': 'run034-sample'}]
        return []
    monkeypatch.setattr(module.cloud, 'cli', cli)
    module.retrieve_and_delete('sample')
    assert calls == [('pod', 'list', '--all'), ('pod', 'delete', 'owned123'), ('pod', 'list', '--all')]
    assert json.loads(target.read_text())['pod_deleted'] is True


def configure_transfer(records, monkeypatch, remote_sha):
    monkeypatch.setattr(module, 'remote_status', lambda label: {'conditions': [{'verified': True, 'exit_code': '0'}]})
    monkeypatch.setattr(module.cloud, 'connect', lambda label: nullcontext(None))
    monkeypatch.setattr(module.cloud, 'command', lambda *args, **kwargs: remote_sha + '  archive.tar')
    module.cloud.write(records / 'ssh-sample.json', {'ip': '127.0.0.1', 'port': 22, 'ssh_key': {'path': 'unused'}})


def test_archive_hash_mismatch_prevents_deletion(setup, monkeypatch):
    records, calls = setup
    archive = records / 'retrieval/sample/results.tar'
    archive.parent.mkdir(parents=True)
    archive.write_bytes(b'bad transfer')
    configure_transfer(records, monkeypatch, '0' * 64)
    monkeypatch.setattr(module.subprocess, 'run', lambda *args, **kwargs: SimpleNamespace(returncode=0))
    with pytest.raises(AssertionError, match='hash mismatch'):
        module.retrieve_and_delete('sample')
    assert calls == []


def test_local_scientific_verification_failure_prevents_deletion(setup, monkeypatch):
    records, calls = setup
    local = records / 'retrieval/sample'
    local.mkdir(parents=True)
    source = local / 'file.json'
    source.write_text('{}')
    archive = local / 'results.tar'
    with tarfile.open(archive, 'w') as tar:
        tar.add(source, arcname='artifacts/example.json')
    configure_transfer(records, monkeypatch, module.digest(archive))
    monkeypatch.setattr(module.subprocess, 'run', lambda *args, **kwargs: SimpleNamespace(returncode=1, stdout='', stderr='invalid coverage'))
    with pytest.raises(RuntimeError, match='scientific verification failed'):
        module.retrieve_and_delete('sample')
    assert calls == []
    assert not (local / 'receipt.json').exists()


def test_bad_incremental_checkpoint_is_never_published(setup, monkeypatch):
    records, calls = setup
    relative = 'artifacts/attempts/001-example/checkpoints/step_000032/model.safetensors'
    row = {'path': relative, 'bytes': 4, 'sha256': hashlib.sha256(b'good').hexdigest()}
    module.cloud.write(records / 'ssh-sample.json', {'ip': '127.0.0.1', 'port': 22, 'ssh_key': {'path': 'unused'}})
    monkeypatch.setattr(module.cloud, 'connect', lambda label: nullcontext(None))
    monkeypatch.setattr(module.cloud, 'command', lambda *args, **kwargs: json.dumps([row]))
    def transfer(command, **kwargs):
        Path(command[-1]).write_bytes(b'bad!')
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(module.subprocess, 'run', transfer)
    with pytest.raises(AssertionError, match='Checkpoint transfer hash mismatch'):
        sync_function('sample')
    assert not (module.HERE / relative).exists()
    assert not (records / 'retrieval/sample/checkpoint-files.json').exists()
    assert calls == []

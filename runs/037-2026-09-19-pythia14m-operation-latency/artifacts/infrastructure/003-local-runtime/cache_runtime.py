"""Infrastructure-only cache relocation, between completed benchmark processes."""
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import signal
import time

RUN = Path('/workspace/run037')
OUT = RUN / 'artifacts/infrastructure/003-local-runtime'
CONTROLLER = 606


def copy_verified(item):
    source, destination, relative = item
    digest = hashlib.sha256()
    with source.open('rb') as src, destination.open('xb') as dst:
        while block := src.read(4 * 1024 * 1024):
            digest.update(block)
            dst.write(block)
    os.chmod(destination, source.stat().st_mode)
    actual = hashlib.sha256()
    with destination.open('rb') as dst:
        while block := dst.read(4 * 1024 * 1024):
            actual.update(block)
    assert actual.digest() == digest.digest(), relative
    return {'path': relative, 'bytes': destination.stat().st_size,
            'sha256': actual.hexdigest()}


def stage(source, destination):
    assert source.is_dir() and not source.is_symlink() and not destination.exists()
    destination.mkdir(parents=True)
    items, links = [], []
    for root, directories, files in os.walk(source):
        base = Path(root)
        target = destination / base.relative_to(source)
        for name in directories + files:
            src, dst = base / name, target / name
            relative = src.relative_to(source).as_posix()
            if src.is_symlink():
                link = os.readlink(src)
                dst.symlink_to(link)
                links.append({'path': relative, 'target': link})
            elif src.is_dir():
                dst.mkdir()
            else:
                items.append((src, dst, relative))
    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
        records = list(pool.map(copy_verified, items))
    return {'source': str(source), 'destination': str(destination),
            'files': records, 'symlinks': links}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    command = Path(f'/proc/{CONTROLLER}/cmdline').read_bytes()
    assert b'03_execute.py' in command and b'smoke' in command
    os.kill(CONTROLLER, signal.SIGSTOP)
    print('Controller paused; letting its active child finish unchanged', flush=True)
    try:
        deadline = time.monotonic() + 240
        while True:
            children = Path(f'/proc/{CONTROLLER}/task/{CONTROLLER}/children').read_text().split()
            active = []
            for child in children:
                status = Path(f'/proc/{child}/status')
                if status.exists() and '\nState:\tZ' not in status.read_text():
                    active.append(child)
            if not active:
                break
            if time.monotonic() > deadline:
                raise RuntimeError('Current child did not finish within relocation window')
            time.sleep(2)
        assert not list((RUN/'artifacts/attempts').glob('scientific-*'))
        roots = [(RUN/'runtime/venv', Path('/tmp/run037-local/venv')),
                 (Path('/workspace/run037-extensions'), Path('/tmp/run037-local/extensions'))]
        report = {'started_epoch': time.time(), 'reason': 'FUSE network-filesystem startup delay',
                  'scientific_inputs_changed': False, 'trees': []}
        for source, destination in roots:
            print(f'Copying and hashing {source}', flush=True)
            report['trees'].append(stage(source, destination))
        for source, destination in roots:
            backup = source.with_name(source.name + '-network-original')
            assert not backup.exists()
            source.rename(backup)
            source.symlink_to(destination, target_is_directory=True)
        report['completed_epoch'] = time.time()
        (OUT/'verified-copy.json').write_text(json.dumps(report, indent=2)+'\n')
        print('Identical runtime/cache copies installed; logical paths preserved', flush=True)
    finally:
        os.kill(CONTROLLER, signal.SIGCONT)
        print('Controller resumed', flush=True)


if __name__ == '__main__':
    main()

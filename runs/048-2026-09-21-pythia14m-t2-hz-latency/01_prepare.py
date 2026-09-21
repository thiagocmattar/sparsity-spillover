"""Freeze the Run041 endpoint and Run037 independent h/z execution controls."""
import argparse
from pathlib import Path
import shutil
import tarfile
from io_utils import RUN, REPO, fs, read, write, record, verify, sha

TRAIN = REPO/'runs/041-2026-09-20-pythia14m-hz-h-only-ol1/latency'
CONTROL = REPO/'runs/037-2026-09-19-pythia14m-operation-latency'


def prepare():
    if (RUN/'artifacts/attempts').exists():
        raise RuntimeError('Executed inputs are immutable')
    copies = []

    def copy(source, relative, expected=None, root=TRAIN):
        if expected is not None:
            assert record(source, root) == expected, source
        target = RUN/relative
        fs(target.parent).mkdir(parents=True, exist_ok=True)
        shutil.copyfile(fs(source), fs(target))
        copies.append({'source': record(source, REPO), 'copy': record(target)})

    archive = read(TRAIN/'provenance/archive.json')
    for row in archive['files']:
        item = row['snapshot']
        copy(TRAIN/item['path'], item['path'], item)
    write(RUN/'provenance/archive.json', archive)
    inputs = read(TRAIN/'provenance/inputs.json')
    checkpoint = next(c for c in inputs['checkpoints'] if c['dose'] == .1)
    assert checkpoint['id'] == 'c03'
    assert checkpoint['source_condition']['active_sites'] == ['h', 'z']
    assert checkpoint['source_condition']['pressure_sites'] == ['h']
    for item in checkpoint['files'] + checkpoint['provenance'] + [inputs['validation']]:
        copy(TRAIN/item['path'], item['path'], item)
    write(RUN/'provenance/inputs.json', {**inputs, 'checkpoints': [checkpoint]})
    for name in ['hz_adapter.py', 'run041_topology.py']:
        copy(TRAIN/name, name)
    copy(TRAIN/'replay.py', 'frozen_replay.py')
    copy(TRAIN/'provenance/pip-freeze.txt', 'provenance/pip-freeze.txt')
    catalog = read(TRAIN/'provenance/candidates.json')
    catalog['configurations'] = [c for c in catalog['configurations'] if c['id'] == 'k050']
    assert len(catalog['configurations']) == 1
    assert catalog['configurations'][0]['settings']['shortcut'] is False
    write(RUN/'provenance/candidates.json', catalog)
    copy(CONTROL/'candidate/joint.cu', 'candidate/joint.cu', root=CONTROL)
    write(RUN/'provenance/reuse.json', {'files': copies})
    verify_all()


def verify_all():
    rows = [r['snapshot'] for r in read(RUN/'provenance/archive.json')['files']]
    inputs = read(RUN/'provenance/inputs.json')
    rows += [inputs['validation']]
    rows += [r for c in inputs['checkpoints'] for r in c['files'] + c['provenance']]
    rows += [r['copy'] for r in read(RUN/'provenance/reuse.json')['files']]
    for row in rows:
        verify(row)
    print(f'Verified {len(rows)} frozen input/source records', flush=True)


def bundle():
    verify_all()
    paths = list(RUN.glob('*.py')) + list(RUN.glob('*.sh')) + [RUN/'config.json']
    paths += list((RUN/'candidate').glob('*'))
    paths += [RUN/r['copy']['path'] for r in read(RUN/'provenance/reuse.json')['files']]
    paths += list((RUN/'provenance').rglob('*'))
    paths = sorted({p for p in paths if fs(p).is_file()})
    target = RUN/'bundles/input-001.tar.gz'
    fs(target.parent).mkdir(exist_ok=True)
    if target.exists():
        raise FileExistsError(target)
    write(RUN/'bundles/inventory-001.json', {'files': [record(p) for p in paths]})
    with tarfile.open(target, 'w:gz', compresslevel=1) as tar:
        for path in paths:
            tar.add(fs(path), arcname=path.relative_to(RUN).as_posix(), recursive=False)
    write(RUN/'bundles/receipt-001.json', record(target))
    print(record(target))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare', 'verify', 'bundle'])
    args = parser.parse_args()
    {'prepare': prepare, 'verify': verify_all, 'bundle': bundle}[args.action]()

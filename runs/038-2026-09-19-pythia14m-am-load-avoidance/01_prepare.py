"""Freeze Run029 c30 inputs and prepare narrowly modified execution controls."""
import argparse
import json
from pathlib import Path
import shutil
import tarfile
from io_utils import RUN, REPO, fs, read, write, record, verify, sha

BASE = REPO / 'runs/037-2026-09-19-pythia14m-operation-latency'


def replace(text, old, new, count=1):
    if text.count(old) != count:
        raise ValueError(f'Frozen-source replacement mismatch: {old[:90]}')
    return text.replace(old, new)


def prepare():
    if (RUN / 'artifacts/attempts').exists():
        raise RuntimeError('Inputs are immutable once execution begins')
    copies = []

    def copy(source, relative, expected=None):
        if expected and record(source, BASE) != expected:
            raise ValueError(f'Source identity mismatch: {source}')
        dest = RUN / relative
        if not dest.resolve().is_relative_to(RUN):
            raise ValueError('Copy escapes run')
        fs(dest.parent).mkdir(parents=True, exist_ok=True)
        if not fs(dest).exists() or sha(dest) != sha(source):
            shutil.copyfile(fs(source), fs(dest))
        copies.append({'source': record(source, REPO), 'copy': record(dest)})

    archived = read(BASE / 'provenance/archive.json')
    for row in archived['files']:
        item = row['snapshot']
        copy(BASE / item['path'], item['path'], item)
    write(RUN / 'provenance/archive.json', archived)
    original = read(BASE / 'provenance/inputs.json')
    endpoint = next(r for r in original['checkpoints'] if r['id'] == 'c30')
    for row in endpoint['files'] + endpoint['provenance'] + [original['validation']]:
        copy(BASE / row['path'], row['path'], row)
    weight = next(r for r in endpoint['files'] if r['path'].endswith('/model.safetensors'))
    assert weight['sha256'] == 'f83aef36ddbddbd94efb915d574c4645c687206bb68f51886ffc6e979985b425'
    development = read(REPO/'runs/027-2026-09-06-pythia14m-kernel-sparsity-characterization/prelaunch/inputs.json')['inputs']['development']
    source = REPO/development['path']
    if record(source, REPO) != development: raise ValueError('Development cache identity mismatch')
    copy(source, 'inputs/development.int32.bin')
    write(RUN / 'provenance/inputs.json', {**original, 'checkpoints': [endpoint],
          'development':record(RUN/'inputs/development.int32.bin')})
    copy(BASE / 'provenance/pip-freeze.txt', 'provenance/pip-freeze.txt')
    catalog = read(BASE / 'provenance/candidates.json')
    catalog['configurations'] = [r for r in catalog['configurations'] if r['id'] == 'k050']
    assert len(catalog['configurations']) == 1
    assert catalog['configurations'][0]['settings']['shortcut'] is False
    write(RUN / 'provenance/candidates.json', catalog)
    write(RUN / 'provenance/reuse.json', {'base_run': BASE.relative_to(REPO).as_posix(), 'files': copies})
    sources = [RUN/'candidate/projection.cu', RUN/'site_port.py', RUN/'diagnostics.py']
    write(RUN/'provenance/control-sources.json', {'files': [record(p) for p in sources]})
    verify_inputs()


def verify_inputs():
    manifest = read(RUN / 'provenance/inputs.json')
    rows = [manifest['validation'],manifest['development']]
    rows += [r['snapshot'] for r in read(RUN / 'provenance/archive.json')['files']]
    rows += [f for c in manifest['checkpoints'] for f in c['files'] + c['provenance']]
    for row in rows:
        verify(row)
    for row in read(RUN / 'provenance/control-sources.json')['files']: verify(row)
    print(f'Verified {len(rows)} frozen source/input identities and derived control sources')


def bundle(tag):
    verify_inputs()
    if not tag.isalnum():
        raise ValueError('Alphanumeric bundle tag required')
    target = RUN / 'bundles' / f'input-{tag}.tar'
    target.parent.mkdir(exist_ok=True)
    if target.exists():
        raise FileExistsError('Use a new bundle tag')
    paths = list(RUN.glob('*.py')) + list(RUN.glob('*.sh')) + [RUN / 'config.json']
    paths += list((RUN / 'candidate').iterdir()) + list((RUN / 'provenance').glob('*'))
    paths += [RUN / r['copy']['path'] for r in read(RUN / 'provenance/reuse.json')['files']]
    paths = sorted({p for p in paths if fs(p).is_file()})
    write(target.with_name(f'input-{tag}-inventory.json'), {'files': [record(p) for p in paths]})
    with tarfile.open(target, 'w') as tar:
        for path in paths:
            tar.add(fs(path), arcname=path.relative_to(RUN).as_posix(), recursive=False)
    write(target.with_suffix('.receipt.json'), record(target))
    print(json.dumps(record(target)))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['prepare', 'verify', 'bundle'])
    p.add_argument('--tag', default='001')
    args = p.parse_args()
    if args.action == 'prepare': prepare()
    elif args.action == 'verify': verify_inputs()
    else: bundle(args.tag)

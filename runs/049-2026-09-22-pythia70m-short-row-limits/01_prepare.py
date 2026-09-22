"""Copy and hash only retained inputs needed by this approved comparison."""
import argparse
import copy
import shutil
import tarfile
from io_utils import RUN, REPO, fs, read, write, record, verify, sha

R45 = REPO / 'runs/045-2026-09-20-pythia70m-kernel-grid'
R42 = REPO / 'runs/042-2026-09-20-pythia70m-sparse-scale'
CANDIDATES = ('opt001', 'opt008', 'opt019', 'opt024', 'opt025', 'opt032', 'opt063', 'opt073')


def prepare():
    if (RUN / 'artifacts/attempts').exists():
        raise RuntimeError('Executed inputs are immutable')
    copies = []

    def keep(source, target):
        dest = RUN / target
        fs(dest.parent).mkdir(parents=True, exist_ok=True)
        if not fs(dest).exists() or sha(dest) != sha(source):
            shutil.copyfile(fs(source), fs(dest))
        copies.append({'source': record(source, REPO), 'copy': record(dest)})
        return record(dest)

    for folder in ('archive', 'kernel', 'base70', *('candidates/' + c for c in CANDIDATES)):
        for source in sorted(fs(R45 / folder).rglob('*')):
            if source.is_file() and '__pycache__' not in source.parts:
                relative = source.relative_to(fs(R45)).as_posix()
                keep(R45 / relative, relative)
    for name in ('archive.json', 'pip-freeze.txt', 'frozen-port.json', 'final-selection.json'):
        keep(R45 / 'provenance' / name, 'provenance/' + name)
    original = read(R45 / 'provenance/inputs.json')
    old_dev = read(R42 / 'provenance/inputs.json')['development']
    manifest = {'validation_metadata': copy.deepcopy(original['validation_metadata']),
                'validation': keep(verify(original['validation'], R45), 'inputs/validation.int32.bin'),
                'development': keep(verify(old_dev, R42), 'inputs/development.int32.bin'),
                'development_selection': 'first16 blocks of retained training development cache',
                'checkpoints': []}
    for cid in read(RUN / 'config.json')['conditions']:
        old = next(r for r in original['checkpoints'] if r['id'] == cid)
        row = copy.deepcopy(old)
        row['checkpoint'] = 'inputs/checkpoints/' + cid
        row['files'] = [keep(verify(item, R45), row['checkpoint'] + '/' + PathName(item)) for item in old['files']]
        row['provenance'] = [keep(verify(item, R45), f'provenance/originals/{cid}/{i:02d}-' + PathName(item))
                             for i, item in enumerate(old['provenance'])]
        manifest['checkpoints'].append(row)
    write(RUN / 'provenance/inputs.json', manifest)
    write(RUN / 'provenance/reuse.json', {'files': copies})
    verify_inputs()


def PathName(item):
    return item['path'].split('/')[-1]


def verify_inputs():
    rows = read(RUN / 'provenance/reuse.json')['files']
    for row in rows:
        verify(row['copy'])
    assert [r['id'] for r in read(RUN / 'provenance/inputs.json')['checkpoints']] == read(RUN / 'config.json')['conditions']
    print(f'Verified {len(rows)} retained files', flush=True)


def freeze():
    verify_inputs()
    paths = list(RUN.glob('*.py')) + list(RUN.glob('*.sh')) + [RUN / 'config.json', RUN / 'short_rows.cu']
    paths += [p for folder in ('kernel', 'base70', 'candidates') for p in (RUN / folder).rglob('*')
              if p.is_file() and '__pycache__' not in p.parts]
    write(RUN / 'provenance/source-freeze.json', {'files': [record(p) for p in sorted(set(paths))]})


def bundle():
    verify_inputs()
    for row in read(RUN / 'provenance/source-freeze.json')['files']:
        verify(row)
    paths = list(RUN.glob('*.py')) + list(RUN.glob('*.sh')) + [RUN / 'config.json', RUN / 'README.md', RUN / 'short_rows.cu']
    paths += [RUN / row['copy']['path'] for row in read(RUN / 'provenance/reuse.json')['files']]
    paths += [p for folder in ('provenance', 'kernel', 'base70', 'candidates') for p in (RUN / folder).rglob('*')
              if p.is_file() and '__pycache__' not in p.parts]
    paths = sorted(set(paths))
    target = RUN / 'bundles/input-001.tar.gz'
    target.parent.mkdir(exist_ok=True)
    if target.exists():
        raise FileExistsError(target)
    write(target.with_name('inventory-001.json'), {'files': [record(p) for p in paths]})
    with tarfile.open(target, 'w:gz', compresslevel=1) as archive:
        for path in paths:
            archive.add(fs(path), arcname=path.relative_to(RUN).as_posix(), recursive=False)
    write(target.with_name('receipt-001.json'), record(target))
    print(record(target), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('prepare', 'verify', 'freeze', 'bundle'))
    args = parser.parse_args()
    {'prepare': prepare, 'verify': verify_inputs, 'freeze': freeze, 'bundle': bundle}[args.action]()

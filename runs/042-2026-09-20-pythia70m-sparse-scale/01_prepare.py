"""Preserve exact prior inputs and add the predeclared same-GPU scale controls."""
import argparse
import copy
import shutil
import tarfile

from io_utils import RUN, REPO, fs, read, write, record, verify, sha


def prepare():
    if (RUN / 'artifacts/attempts').exists():
        raise RuntimeError('Inputs immutable after execution')
    old = REPO / 'runs/040-2026-09-20-pythia70m-kernel-overhead'
    copies = []

    def keep(source, target):
        dest = RUN / target
        fs(dest.parent).mkdir(parents=True, exist_ok=True)
        if not fs(dest).exists() or sha(dest) != sha(source):
            shutil.copyfile(fs(source), fs(dest))
        copies.append({'source': record(source, REPO), 'copy': record(dest)})
        return record(dest)

    for row in read(old / 'provenance/reuse.json')['files']:
        verify(row['copy'], old)
        keep(old / row['copy']['path'], row['copy']['path'])
    for name in ('archive.json', 'frozen-port.json', 'pip-freeze.txt'):
        keep(old / 'provenance' / name, 'provenance/' + name)
    manifest = read(old / 'provenance/inputs.json')
    r35 = REPO / 'runs/035-2026-09-18-pythia70m-k050-port'
    added = next(c for c in read(r35 / 'provenance/inputs.json')['checkpoints'] if c['id'] == 'c16')
    for item in added['files'] + added['provenance']:
        verify(item, r35)
        keep(r35 / item['path'], item['path'])
    manifest['checkpoints'].append(added)
    r29 = REPO / 'runs/029-2026-09-07-pythia14m-matched-kernel-retrospective'
    source = read(r29 / 'provenance/inputs.json')
    assert source['validation']['sha256'] == manifest['validation']['sha256']
    for original in source['checkpoints']:
        if original['id'] not in ('c01', 'c20', 'c35'):
            continue
        row = copy.deepcopy(original)
        row['source_id'] = row['id']
        row['id'] = 'm14-' + row['id']
        row['checkpoint'] = 'inputs14m/' + row['checkpoint']
        for field in ('files', 'provenance'):
            row[field] = [keep(verify(item, r29), 'inputs14m/' + item['path']) for item in original[field]]
        manifest['checkpoints'].append(row)
    write(RUN / 'provenance/inputs.json', manifest)
    write(RUN / 'provenance/reuse.json', {'files': copies})
    verify_inputs()


def verify_inputs():
    items = read(RUN / 'provenance/reuse.json')['files']
    for item in items:
        verify(item['copy'])
    print('Verified', len(items), 'retained input/source copies', flush=True)


def bundle(tag):
    verify_inputs()
    target = RUN / 'bundles' / f'input-{tag}.tar.gz'
    target.parent.mkdir(exist_ok=True)
    if target.exists():
        raise FileExistsError(target)
    paths = list(RUN.glob('*.py')) + list(RUN.glob('*.sh')) + [RUN/'config.json', RUN/'README.md']
    paths += [RUN / row['copy']['path'] for row in read(RUN/'provenance/reuse.json')['files']]
    for folder in ('provenance', 'kernel', 'base70', 'candidates'):
        paths += [p for p in (RUN/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    paths = sorted(set(paths))
    write(target.with_name(f'input-{tag}-inventory.json'), {'files': [record(p) for p in paths]})
    with tarfile.open(target, 'w:gz', compresslevel=1) as archive:
        for path in paths:
            archive.add(fs(path), arcname=path.relative_to(RUN).as_posix(), recursive=False)
    write(target.with_name(f'input-{tag}-receipt.json'), record(target))
    print(record(target), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('prepare', 'verify', 'bundle'))
    parser.add_argument('--tag', default='001')
    args = parser.parse_args()
    if args.action == 'prepare': prepare()
    elif args.action == 'verify': verify_inputs()
    else: bundle(args.tag)

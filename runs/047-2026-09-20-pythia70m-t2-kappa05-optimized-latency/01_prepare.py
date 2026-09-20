"""Freeze the existing implementation and three approved final checkpoints."""
import argparse
import copy
import shutil
import tarfile
from io_utils import RUN, REPO, fs, read, write, record, verify, sha

R45 = REPO/'runs/045-2026-09-20-pythia70m-kernel-grid'
R46 = REPO/'runs/046-2026-09-20-pythia70m-hz-h-only-ol1-kappa05/latency'
CANDIDATES = ('opt001', 'opt008', 'opt019', 'opt024', 'opt025', 'opt032', 'opt063', 'opt073')


def prepare():
    if (RUN/'artifacts/attempts').exists():
        raise RuntimeError('Executed inputs are immutable')
    copies = []

    def keep(source, target):
        dest = RUN/target
        fs(dest.parent).mkdir(parents=True, exist_ok=True)
        if not fs(dest).exists() or sha(dest) != sha(source):
            shutil.copyfile(fs(source), fs(dest))
        copies.append({'source': record(source, REPO), 'copy': record(dest)})
        return record(dest)

    for folder in ('archive', 'kernel', 'base70', *('candidates/'+c for c in CANDIDATES)):
        for source in sorted(fs(R45/folder).rglob('*')):
            if source.is_file() and '__pycache__' not in source.parts:
                relative = source.relative_to(fs(R45)).as_posix()
                keep(R45/relative, relative)
    for name in ('archive.json', 'pip-freeze.txt', 'frozen-port.json', 'final-selection.json'):
        keep(R45/'provenance'/name, 'provenance/'+name)
    for name in ('hz_adapter.py', 'hz_topology.py', 'frozen_replay.py'):
        keep(R45/name, name)

    original = read(R45/'provenance/inputs.json')
    later = read(R46/'provenance/inputs.json')
    if original['validation']['sha256'] != later['validation']['sha256']:
        raise ValueError('Validation cache mismatch')
    manifest = {key: copy.deepcopy(original[key]) for key in ('validation', 'validation_metadata')}
    manifest['validation'] = keep(verify(original['validation'], R45), 'inputs/validation.int32.bin')
    manifest['checkpoints'] = []
    selections = [(R45, next(r for r in original['checkpoints'] if r['id']==cid), cid)
                  for cid in ('c00', 'c25')]
    point, = later['checkpoints']
    assert point['family']=='HZ+OL1@h' and point['dose']==.5
    assert point['final_checkpoint_content_sha256']=='d6a3813f83ce080b07637fe422bb7ab1b23a6793e39c365601301a661851d02f'
    selections.append((R46, point, 'c26'))
    for source, old_row, cid in selections:
        row = copy.deepcopy(old_row)
        row.update(source_id=old_row['id'], source_run=source.relative_to(REPO).as_posix(),
                   id=cid, checkpoint=f'inputs/checkpoints/{cid}')
        row['files'] = [keep(verify(item, source), row['checkpoint']+'/'+item['path'].split('/')[-1])
                        for item in old_row['files']]
        row['provenance'] = [keep(verify(item, source), f"provenance/originals/{cid}/{i:02d}-"+item['path'].split('/')[-1])
                            for i, item in enumerate(old_row['provenance'])]
        manifest['checkpoints'].append(row)
    write(RUN/'provenance/inputs.json', manifest)
    write(RUN/'provenance/reuse.json', {'files': copies,
          'candidate_policy': 'Byte-identical Run045 opt073 and original port; no new search or arithmetic changes.',
          'new_endpoint': 'c26; Run046 T2/Ph kappa=.5',
          'session_references': ['c00: Base', 'c25: T2/Ph kappa=.1']})
    verify_inputs()


def verify_inputs():
    items = [row['copy'] for row in read(RUN/'provenance/reuse.json')['files']]
    for item in items:
        verify(item)
    inputs = read(RUN/'provenance/inputs.json')
    assert [r['id'] for r in inputs['checkpoints']]==read(RUN/'config.json')['conditions']
    for row in inputs['checkpoints']:
        for item in row['files'] + row['provenance']:
            verify(item)
    verify(inputs['validation'])
    selection = read(RUN/'provenance/final-selection.json')
    verify(selection['manifest'])
    for candidate in CANDIDATES:
        for row in read(RUN/'candidates'/candidate/'manifest.json')['files']:
            verify(row)
    print(f'Verified {len(items)} retained source/input files and three checkpoint identities', flush=True)


def freeze():
    verify_inputs()
    paths = list(RUN.glob('*.py')) + list(RUN.glob('*.sh')) + [RUN/'config.json']
    paths += [p for folder in ('kernel', 'base70', 'candidates') for p in (RUN/folder).rglob('*')
              if p.is_file() and '__pycache__' not in p.parts]
    write(RUN/'provenance/source-freeze.json', {'files': [record(p) for p in sorted(set(paths))]})


def bundle(tag):
    verify_inputs()
    for row in read(RUN/'provenance/source-freeze.json')['files']:
        verify(row)
    paths = list(RUN.glob('*.py')) + list(RUN.glob('*.sh')) + [RUN/'config.json', RUN/'README.md']
    paths += [RUN/row['copy']['path'] for row in read(RUN/'provenance/reuse.json')['files']]
    paths += [p for folder in ('provenance', 'kernel', 'base70', 'candidates') for p in (RUN/folder).rglob('*')
              if p.is_file() and '__pycache__' not in p.parts]
    paths = sorted(set(paths))
    target = RUN/'bundles'/f'input-{tag}.tar.gz'
    target.parent.mkdir(exist_ok=True)
    if target.exists():
        raise FileExistsError(target)
    write(target.with_name(f'inventory-{tag}.json'), {'files': [record(p) for p in paths]})
    with tarfile.open(target, 'w:gz', compresslevel=1) as archive:
        for path in paths:
            archive.add(fs(path), arcname=path.relative_to(RUN).as_posix(), recursive=False)
    write(target.with_name(f'receipt-{tag}.json'), record(target))
    print(record(target), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('prepare', 'verify', 'freeze', 'bundle'))
    parser.add_argument('--tag', default='001')
    args = parser.parse_args()
    if args.action == 'bundle':
        bundle(args.tag)
    else:
        {'prepare': prepare, 'verify': verify_inputs, 'freeze': freeze}[args.action]()

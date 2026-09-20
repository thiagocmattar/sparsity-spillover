"""Hash-preserve frozen implementations and attach the 26 approved endpoints."""
import argparse
import copy
import shutil
import tarfile
from io_utils import RUN, REPO, fs, read, write, record, verify, sha

R35 = REPO/'runs/035-2026-09-18-pythia70m-k050-port'
R42 = REPO/'runs/042-2026-09-20-pythia70m-sparse-scale'
R43 = REPO/'runs/043-2026-09-20-pythia70m-hz-h-only-ol1/latency'
# Frozen dependency closure, including the parent provenance manifests.
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
        for source in sorted(fs(R42/folder).rglob('*')):
            if source.is_file() and '__pycache__' not in source.parts:
                relative = source.relative_to(fs(R42)).as_posix()
                keep(R42/relative, relative)
    for name in ('archive.json', 'pip-freeze.txt', 'frozen-port.json', 'final-selection.json'):
        keep(R42/'provenance'/name, 'provenance/'+name)
    keep(R43/'hz_adapter.py', 'hz_adapter.py')
    keep(R43/'run043_topology.py', 'hz_topology.py')

    # Only redirect the adapter import. The frozen candidate's operators are unchanged.
    target = RUN/'kernel/candidate.py'
    original = fs(target).read_text(encoding='utf-8')
    old = "original=module('run035_frozen_adapter',replay.R27/'adapter.py')"
    new = "original=module('run035_frozen_adapter',HERE.parent/'hz_adapter.py')"
    if original.count(old) != 1:
        raise ValueError('Unexpected frozen composition source')
    fs(target).write_text(original.replace(old, new), encoding='utf-8', newline='\n')
    bridge = {'source': record(R42/'kernel/candidate.py', REPO), 'copy': record(target),
              'change': {'before': old, 'after': new}, 'arithmetic_changed': False}
    copies = [row for row in copies if row['copy']['path'] != 'kernel/candidate.py']
    text = fs(R42/'frozen_replay.py').read_text(encoding='utf-8')
    text += "\nmodule('run045_hz_topology', RUN/'hz_topology.py').register()\n"
    fs(RUN/'frozen_replay.py').write_text(text, encoding='utf-8', newline='\n')

    original = read(R35/'provenance/inputs.json')
    hz = read(R43/'provenance/inputs.json')
    if original['validation']['sha256'] != hz['validation']['sha256']:
        raise ValueError('Validation cache mismatch')
    manifest = {key: copy.deepcopy(original[key]) for key in ('validation', 'validation_metadata')}
    manifest['validation'] = keep(verify(original['validation'], R35), 'inputs/validation.int32.bin')
    manifest['checkpoints'] = []
    for source, rows in ((R35, original['checkpoints']), (R43, hz['checkpoints'])):
        for old_row in rows:
            row = copy.deepcopy(old_row)
            row['source_id'] = old_row['id']
            row['source_run'] = source.relative_to(REPO).as_posix()
            row['id'] = f"c{len(manifest['checkpoints']):02d}"
            row['checkpoint'] = f"inputs/checkpoints/{row['id']}"
            row['files'] = [keep(verify(item, source), row['checkpoint']+'/'+item['path'].split('/')[-1])
                            for item in old_row['files']]
            row['provenance'] = [keep(verify(item, source), f"provenance/originals/{row['id']}/{i:02d}-"+item['path'].split('/')[-1])
                                 for i, item in enumerate(old_row['provenance'])]
            manifest['checkpoints'].append(row)
    if len(manifest['checkpoints']) != 26:
        raise ValueError('Expected 22 original plus four HZ checkpoints')
    write(RUN/'provenance/inputs.json', manifest)
    write(RUN/'provenance/reuse.json', {'files': copies, 'compatibility_bridge': bridge,
          'candidate_policy': 'Frozen opt073; no search or checkpoint-specific selection'})
    verify_inputs()


def verify_inputs():
    reuse = read(RUN/'provenance/reuse.json')
    items = [row['copy'] for row in reuse['files']] + [reuse['compatibility_bridge']['copy']]
    for item in items:
        verify(item)
    inputs = read(RUN/'provenance/inputs.json')
    for row in inputs['checkpoints']:
        for item in row['files'] + row['provenance']:
            verify(item)
    verify(inputs['validation'])
    selection = read(RUN/'provenance/final-selection.json')
    verify(selection['manifest'])
    for candidate in CANDIDATES:
        for row in read(RUN/'candidates'/candidate/'manifest.json')['files']:
            verify(row)
    print(f'Verified {len(items)} retained source/input files and 26 checkpoint identities', flush=True)


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

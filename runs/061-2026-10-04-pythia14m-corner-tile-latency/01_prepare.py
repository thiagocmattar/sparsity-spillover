"""Copy verified frozen consumers, native corner operator and four final models."""
import argparse
import shutil
import tarfile
from io_utils import RUN, REPO, fs, read, write, record, verify

BASE = REPO/'runs/059-2026-10-04-pythia14m-p2-ph-latency'
SCALAR = REPO/'runs/058-2026-10-03-pythia14m-t2-p2-versus-ph'
TILE = REPO/'runs/060-2026-10-04-pythia14m-temporal-corner-tile'


def prepare():
    if (RUN/'artifacts/attempts').exists():
        raise RuntimeError('Inputs are immutable after execution begins')
    copies = []

    def copy(source, relative, expected=None):
        if expected is not None and record(source, source.parents[0])['sha256'] != expected['sha256']:
            raise ValueError('Source SHA mismatch: '+str(source))
        target = RUN/relative
        if not target.resolve().is_relative_to(RUN):
            raise ValueError('Copy escaped run')
        fs(target.parent).mkdir(parents=True, exist_ok=True)
        shutil.copyfile(fs(source), fs(target))
        copies.append({'source':record(source, REPO), 'copy':record(target)})
        return record(target)

    frozen = read(BASE/'provenance/archive.json')
    for entry in frozen['files']:
        row = entry['snapshot']
        verify(row, BASE)
        copy(BASE/row['path'], row['path'], row)
    write(RUN/'provenance/archive.json', frozen)
    copy(BASE/'hz_adapter.py', 'hz_adapter.py')
    copy(TILE/'tile_gate.py', 'tile_gate.py')
    copy(TILE/'topology.py', 'topology.py')
    copy(BASE/'provenance/pip-freeze.txt', 'provenance/pip-freeze.txt')
    baseline = read(BASE/'provenance/inputs.json')
    verify(baseline['validation'], BASE)
    validation = copy(BASE/baseline['validation']['path'], 'inputs/validation.int32.bin', baseline['validation'])
    checkpoints = []
    for train, family in [(SCALAR,'scalar'), (TILE,'tile')]:
        if read(train/'artifacts/verification.json')['status'] != 'verified':
            raise ValueError('Training cohort not verified')
        for source in sorted((train/'artifacts/attempts').iterdir()):
            if not (source/'manifest.json').is_file():
                continue
            manifest = read(source/'manifest.json')
            c = manifest['condition']
            if c.get('pressure_recipe') != 'ph':
                continue
            if manifest['status'] != 'completed' or manifest['completed_steps'] != 712 or c['pressure_sites'] != ['h'] or c['active_sites'] != ['h','z']:
                raise ValueError('Unexpected source recipe')
            identifier = c['id']
            checkpoint = source/'checkpoints/step_000712'
            names = ['config.json','model.safetensors','checkpoint_metadata.json','generation_config.json']
            files = [copy(checkpoint/name, f'inputs/checkpoints/{identifier}/{name}') for name in names]
            provenance = [copy(source/name, f'provenance/originals/{identifier}/{name}')
                          for name in ['manifest.json','config.yaml','diagnostics/logical_products.json']]
            checkpoints.append({'id':identifier,'family':family,'dose':c['gate_threshold'],
                'checkpoint':f'inputs/checkpoints/{identifier}','files':files,'provenance':provenance,
                'source':source.relative_to(REPO).as_posix(),'source_condition':c,
                'topology':read(checkpoint/'config.json'),
                'original_checkpoint':checkpoint.relative_to(REPO).as_posix(),
                'original_files':[record(checkpoint/name, REPO) for name in names],
                'final_checkpoint_content_sha256':manifest['checkpoints']['final']['content_sha256'],
                'architecture':read(source/'diagnostics/logical_products.json')['architecture_maximum']})
    expected = {identifier for pair in read(RUN/'config.json')['pair_conditions'].values() for identifier in pair}
    if len(checkpoints) != 4 or {c['id'] for c in checkpoints} != expected:
        raise ValueError('Exactly the approved four Ph final checkpoints required')
    write(RUN/'provenance/inputs.json', {'checkpoints':checkpoints,'validation':validation,
          'validation_metadata':baseline['validation_metadata']})
    write(RUN/'provenance/reuse.json', {'files':copies,'frozen_base':BASE.relative_to(REPO).as_posix(),
          'adaptation':'run-local joint.cu and kernel.py; historical consumers and source checkpoints unchanged'})
    print('Prepared four Ph checkpoints and frozen dependencies')


def verify_all():
    inputs = read(RUN/'provenance/inputs.json')
    rows = [inputs['validation']]
    rows += [r['snapshot'] for r in read(RUN/'provenance/archive.json')['files']]
    rows += [f for c in inputs['checkpoints'] for f in c['files']+c['provenance']]
    for row in rows:
        verify(row)
    for row in read(RUN/'provenance/reuse.json')['files']:
        verify(row['copy'])
    print(f'Verified {len(rows)} source/input identities')


def bundle(tag):
    verify_all()
    target = RUN/'bundles'/f'input-{tag}.tar'
    target.parent.mkdir(exist_ok=True)
    if target.exists():
        raise ValueError('Use a fresh bundle tag')
    paths = list(RUN.glob('*.py'))+list(RUN.glob('*.cu'))+list(RUN.glob('*.sh'))+[RUN/'config.json']
    paths += [RUN/r['copy']['path'] for r in read(RUN/'provenance/reuse.json')['files']]
    paths += [p for p in (RUN/'provenance').glob('*') if p.is_file()]
    paths = sorted(set(paths))
    inventory = {'files':[record(p) for p in paths],'bytes':sum(fs(p).stat().st_size for p in paths)}
    write(RUN/'bundles'/f'input-{tag}-inventory.json', inventory)
    with tarfile.open(fs(target),'w') as archive:
        for path in paths:
            archive.add(fs(path),arcname=path.relative_to(RUN).as_posix(),recursive=False)
    write(target.with_suffix('.receipt.json'),record(target))
    print(record(target))


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['prepare','verify','bundle'])
    parser.add_argument('--tag',default='001')
    args=parser.parse_args()
    {'prepare':prepare,'verify':verify_all,'bundle':lambda:bundle(args.tag)}[args.action]()

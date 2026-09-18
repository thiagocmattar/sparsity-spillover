"""Freeze four controls, 40 retained clipping records and original kernels."""
import argparse
import shutil
import tarfile
from pathlib import Path
from io_utils import RUN, REPO, fs, read, write, record, verify


def prepare():
    if (RUN/'artifacts/attempts').exists():
        raise RuntimeError('Inputs are immutable after execution')
    base = REPO/'runs/035-2026-09-18-pythia70m-k050-port'
    copied = []
    def copy(source, relative):
        target = RUN/relative
        fs(target.parent).mkdir(parents=True, exist_ok=True)
        shutil.copyfile(fs(source), fs(target))
        copied.append({'source': record(source, REPO), 'copy': record(target)})
        return record(target)
    archive = read(base/'provenance/archive.json')
    for row in archive['files']:
        item = row['snapshot']
        verify(item, base)
        copy(base/item['path'], item['path'])
    write(RUN/'provenance/archive.json', archive)
    for source in sorted((base/'kernel').rglob('*')):
        if source.is_file() and '__pycache__' not in source.parts:
            copy(source, 'kernel70/'+source.relative_to(base/'kernel').as_posix())
    copy(base/'provenance/pip-freeze.txt', 'provenance/pip-freeze.txt')
    source_manifest = read(base/'provenance/inputs.json')
    validation = copy(base/source_manifest['validation']['path'], 'inputs/validation.int32.bin')
    assert validation['sha256'] == '51cd758fda72f14383da30c358a895d0223c0d1d80b31455d2d842c3656d0451'
    release_path = REPO/'runs/030-2026-09-08-all-models-posthoc-clipping/results/clipping-points.json'
    release = read(release_path)
    paper = read(REPO/'analyses/024-2026-09-17-h-only-kernel-latency/data/paper-checkpoints.json')
    points = [p for p in release['points'] if p['scale'] in ('14M', '70M') and p['family'] in ('A0', 'A1-H')]
    assert len(points) == 40
    checkpoints = []
    for scale, prefix in [('14M', '029-'), ('70M', '035-')]:
        old = next((REPO/'runs').glob(prefix+'*'))
        source = read(old/'provenance/inputs.json')
        for family in ('A0', 'A1-H'):
            c = next(c for c in source['checkpoints'] if c['family'] == family)
            control = next(c for c in paper['checkpoints'] if c['model'] == scale and c['family'] == family)
            assert c['original_checkpoint'] == str(Path(control['source_attempt'])/'checkpoints/step_000712').replace('\\', '/')
            identifier = f'{scale}-{family}'
            files = [copy(old/r['path'], f'inputs/checkpoints/{identifier}/'+Path(r['path']).name) for r in c['files']]
            for row in c['original_files']:
                assert next(f for f in files if Path(f['path']).name == Path(row['path']).name)['sha256'] == row['sha256']
            provenance = [copy(old/r['path'], f'provenance/originals/{identifier}/'+str(Path(r['path']).relative_to(Path(r['path']).parents[0]))) for r in c['provenance']]
            cp = {**c, 'id': identifier, 'scale': scale, 'checkpoint': f'inputs/checkpoints/{identifier}',
                  'files': files, 'provenance': provenance, 'paper_checkpoint_key': control['checkpoint_key']}
            checkpoints.append(cp)
    # The Run030 manifest binds release IDs to exact original checkpoint content.
    clipping_manifest = read(REPO/'runs/030-2026-09-08-all-models-posthoc-clipping/input-manifest.json')
    for cp in checkpoints:
        old = next(c for c in clipping_manifest['checkpoints'] if c['scale']==cp['scale'] and c['family']==cp['family'])
        assert old['checkpoint'] == cp['original_checkpoint']
        assert {Path(r['path']).name:r['sha256'] for r in cp['files']} == {r['name']:r['sha256'] for r in old['files']}
        for point in points:
            if point['scale']==cp['scale'] and point['family']==cp['family']:
                assert point['checkpoint_content_sha256']==old['checkpoint_content_sha256']
    write(RUN/'provenance/clipping-source-manifest.json', {'checkpoints':[
        c for c in clipping_manifest['checkpoints'] if c['scale'] in ('14M','70M') and c['family'] in ('A0','A1-H')]})
    write(RUN/'provenance/clipping-points.json', {'source': record(release_path, REPO), 'points': points,
        'protocol': release['protocol']})
    conditions = []
    for cp in checkpoints:
        grid = sorted([p for p in points if p['scale'] == cp['scale'] and p['family'] == cp['family']], key=lambda p:p['dose'])
        assert [p['dose'] for p in grid] == [i/10 for i in range(10)]
        for point in grid:
            conditions.append({'id': f'c{len(conditions):02d}', 'checkpoint_id': cp['id'],
                'candidate': 'k050' if cp['scale'] == '14M' else 'k050-70m-v2',
                'scale': cp['scale'], 'family': cp['family'], 'p': point['dose'],
                'retained_point': point})
    write(RUN/'provenance/inputs.json', {'checkpoints': checkpoints, 'conditions': conditions,
        'validation': validation, 'validation_metadata': source_manifest['validation_metadata']})
    write(RUN/'provenance/reuse.json', {'files': copied})
    write(RUN/'provenance/candidates.json', {'configurations':[
        {'id':name, 'candidate':name, 'status':'eligible',
         'settings':{'shortcut':False,'round_p':False,'skip':True,'projection_skip':True}}
        for name in ('k050','k050-70m-v2')]})
    print(f'Prepared {len(checkpoints)} controls, {len(conditions)} points and {len(copied)} frozen files')


def verify_all():
    rows = [r['copy'] for r in read(RUN/'provenance/reuse.json')['files']]
    for row in rows:
        verify(row)
    print(f'Verified {len(rows)} frozen file identities')


def bundle():
    verify_all()
    paths = [RUN/r['copy']['path'] for r in read(RUN/'provenance/reuse.json')['files']]
    paths += list(RUN.glob('*.py'))+list(RUN.glob('*.sh'))+[RUN/'config.json']
    paths += [p for p in (RUN/'provenance').glob('*') if p.is_file()]
    paths = sorted(set(paths))
    target = RUN/'bundles/input-001.tar'
    target.parent.mkdir(exist_ok=True)
    if target.exists():
        raise RuntimeError('Do not overwrite a frozen bundle')
    inventory = RUN/'bundles/input-001-inventory.json'
    write(inventory, {'files':[record(p) for p in paths]})
    with tarfile.open(target, 'w') as output:
        for p in paths+[inventory]:
            output.add(fs(p), arcname=p.relative_to(RUN).as_posix(), recursive=False)
    receipt = record(target)
    write(target.with_suffix('.receipt.json'), receipt)
    print(receipt)


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','verify','bundle'])
    action=parser.parse_args().action
    {'prepare':prepare,'verify':verify_all,'bundle':bundle}[action]()

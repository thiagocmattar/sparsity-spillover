"""Package unchanged Run 029 K050 sources with only Run 032's five endpoints."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tarfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BASE = ROOT / "runs/029-2026-09-07-pythia14m-matched-kernel-retrospective"
TRAIN = ROOT / "runs/032-2026-09-16-pythia14m-a7-h-only-ol1"
KAPPAS = [0, .01, .05, .1, .5]


def fs(path):
    path = Path(path).absolute()
    return Path('\\\\?\\' + str(path)) if os.name == 'nt' and not str(path).startswith('\\\\?\\') else path


def sha(path):
    with fs(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def record(path, root=HERE):
    return dict(path=path.relative_to(root).as_posix(), bytes=fs(path).stat().st_size, sha256=sha(path))


def read(path):
    return json.loads(fs(path).read_text(encoding='utf-8'))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n', encoding='utf-8', newline='\n')


def prepare():
    if (HERE/'artifacts/attempts').exists():
        raise RuntimeError('Do not rebuild inputs after execution has begun')
    cohort = read(TRAIN/'artifacts/verification.json')
    assert cohort['status'] == 'verified' and cohort['condition_count'] == 5
    baseline = read(BASE/'provenance/inputs.json')
    archive = read(BASE/'provenance/archive.json')
    copied = []

    def copy(source, relative, expected=None):
        target = HERE/relative
        assert target.resolve().is_relative_to(HERE)
        if expected:
            assert fs(source).stat().st_size == expected['bytes'] and sha(source) == expected['sha256'], source
        fs(target.parent).mkdir(parents=True, exist_ok=True)
        if not fs(target).exists() or sha(target) != sha(source):
            shutil.copyfile(fs(source), fs(target))
        copied.append({'source': record(source, ROOT), 'copy': record(target)})
        return record(target)

    # Existing scientific implementation is frozen byte-for-byte; no kernel port.
    for name in ['02_benchmark.py', 'io_utils.py', 'replay.py']:
        copy(BASE/name, name)
    for row in archive['files']:
        item = row['snapshot']
        copy(BASE/item['path'], item['path'], item)
    write(HERE/'provenance/archive.json', archive)
    catalog = read(BASE/'provenance/candidates.json')
    catalog['configurations'] = [r for r in catalog['configurations'] if r['id'] == 'k050']
    assert len(catalog['configurations']) == 1
    write(HERE/'provenance/candidates.json', catalog)
    copy(BASE/'provenance/pip-freeze.txt', 'provenance/pip-freeze.txt')
    validation = copy(BASE/baseline['validation']['path'], 'inputs/validation.int32.bin', baseline['validation'])
    metadata = baseline['validation_metadata']
    checkpoints = []
    for index, verified in enumerate(sorted(cohort['conditions'], key=lambda r:r['condition']['gate_threshold']), 36):
        condition = verified['condition']
        assert condition['pressure_sites'] == ['h'] and verified['ol1']['pressure_capture_tensor_count'] == 6
        assert verified['completed_steps'] == 712
        source = TRAIN/'artifacts/attempts'/verified['attempt_id']
        checkpoint = source/'checkpoints/step_000712'
        identifier = f'c{index}'
        names = ['config.json', 'model.safetensors', 'checkpoint_metadata.json', 'generation_config.json']
        files = [copy(checkpoint/name, f'inputs/checkpoints/{identifier}/{name}') for name in names]
        provenance = [copy(source/name, f'provenance/originals/{identifier}/{name}')
                      for name in ['manifest.json', 'config.yaml', 'diagnostics/logical_products.json']]
        logical = read(source/'diagnostics/logical_products.json')
        checkpoints.append({
            'id': identifier, 'family': 'A7+OL1@h', 'dose': condition['gate_threshold'],
            'historical': False, 'checkpoint': f'inputs/checkpoints/{identifier}',
            'files': files, 'provenance': provenance, 'original_files': [record(checkpoint/n, ROOT) for n in names],
            'original_checkpoint': checkpoint.relative_to(ROOT).as_posix(),
            'source': source.relative_to(ROOT).as_posix(), 'source_condition': condition,
            'topology': read(checkpoint/'config.json'), 'canonical_logical_products': logical,
            'final_checkpoint_content_sha256': verified['checkpoint_content_sha256'],
        })
    assert [r['dose'] for r in checkpoints] == KAPPAS
    write(HERE/'provenance/inputs.json', {'checkpoints':checkpoints, 'validation':validation,
          'validation_metadata':metadata, 'source_manifest':record(TRAIN/'artifacts/verification.json', ROOT)})
    write(HERE/'provenance/reuse.json', {'base_run':BASE.relative_to(ROOT).as_posix(), 'files':copied})
    print(f'Prepared {len(checkpoints)} new checkpoints and {len(archive["files"])} frozen source files')


def verify():
    inputs = read(HERE/'provenance/inputs.json')
    rows = [inputs['validation']]
    rows += [r['snapshot'] for r in read(HERE/'provenance/archive.json')['files']]
    rows += [f for c in inputs['checkpoints'] for f in c['files']+c['provenance']]
    for row in rows:
        path = HERE/row['path']
        assert path.resolve().is_relative_to(HERE)
        assert record(path) == row, path
    for row in read(HERE/'provenance/reuse.json')['files']:
        assert record(HERE/row['copy']['path']) == row['copy']
    print(f'Verified {len(rows)} source/input identities')


def bundle(tag):
    verify()
    target = HERE/'bundles'/f'input-{tag}.tar'
    target.parent.mkdir(exist_ok=True)
    assert not target.exists(), 'Use a new bundle tag'
    paths = list(HERE.glob('*.py')) + list(HERE.glob('*.sh')) + [HERE/'config.json']
    paths += [HERE/r['copy']['path'] for r in read(HERE/'provenance/reuse.json')['files']]
    paths += list((HERE/'provenance').glob('*'))
    paths = sorted({p for p in paths if fs(p).is_file()})
    inventory = {'files':[record(p) for p in paths], 'bytes':sum(fs(p).stat().st_size for p in paths)}
    write(HERE/'bundles'/f'input-{tag}-inventory.json', inventory)
    with tarfile.open(target, 'w') as archive:
        for path in paths:
            archive.add(fs(path), arcname=path.relative_to(HERE).as_posix(), recursive=False)
    write(target.with_suffix('.receipt.json'), record(target))
    print(record(target))


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','verify','bundle']);parser.add_argument('--tag',default='001')
    args=parser.parse_args()
    if args.action=='prepare':prepare()
    elif args.action=='verify':verify()
    else:bundle(args.tag)

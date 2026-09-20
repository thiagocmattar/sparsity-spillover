"""Record immutable development candidates and freeze one before final timing."""
import argparse
from io_utils import RUN, read, write, record, verify


def register(identifier):
    folder = RUN / 'candidates' / identifier
    manifest_path = folder / 'manifest.json'
    if manifest_path.exists(): raise FileExistsError('Keep candidate immutable; use a new identifier')
    cfg = read(RUN / 'config.json')
    if len(list((RUN / 'candidates').glob('opt*/manifest.json'))) >= cfg['optimization_candidate_budget']:
        raise ValueError('Approved development candidate budget exhausted')
    paths = sorted(p for p in folder.rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    assert folder / 'candidate.py' in paths
    assert all(p.suffix in ('.py', '.cu', '.h', '.cuh', '.md', '.json') for p in paths)
    write(manifest_path, {'candidate': identifier, 'files': [record(p) for p in paths],
          'selection_data': 'fixed first16 training-development blocks only'})


def freeze(identifier):
    destination = RUN / 'provenance/final-selection.json'
    if destination.exists(): raise FileExistsError('Final selection already frozen')
    manifest_path = RUN / 'candidates' / identifier / 'manifest.json'
    manifest = read(manifest_path)
    for item in manifest['files']: verify(item)
    evidence = []
    for path in sorted((RUN / 'artifacts/attempts').glob('*/result.json')):
        result = read(path)
        if result.get('candidate') != identifier: continue
        if not result['arguments'].get('development'):
            raise ValueError('Candidate already exposed to final validation')
        if result['status'] == 'complete' and result.get('qualified'):
            assert result['optimization_candidate'] == manifest
            assert result['evaluation_split'] == 'training-development'
            assert result['validation_blocks'] == read(RUN / 'config.json')['development_blocks']
            evidence.append({'condition': result['condition'], 'result': record(path),
                             'timing': record(path.with_name('timing.json'))})
    assert {e['condition'] for e in evidence} == {'c00', 'c21'}
    write(destination, {'candidate': identifier, 'manifest': record(manifest_path),
          'development_evidence': evidence,
          'criterion': 'Qualified on both checkpoints; selected by T7/Ph development latency. Base latency also reported.'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['register', 'freeze'])
    parser.add_argument('candidate')
    args = parser.parse_args()
    assert args.candidate.startswith('opt') and len(args.candidate) == 6 and args.candidate[3:].isdigit()
    (register if args.action == 'register' else freeze)(args.candidate)

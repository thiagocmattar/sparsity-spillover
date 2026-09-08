"""Verify transferred inputs or a completed diagnostic and inventory its outputs."""
import argparse
from pathlib import Path

from density import GRID, GROUPS, SITES, pool, read_json, sha, validate_row, write_json

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs', action='store_true')
    parser.add_argument('--attempt', type=Path)
    args = parser.parse_args()
    if args.inputs:
        inventory = read_json(HERE/'bundle-inventory.json')
        for relative, entry in inventory.items():
            path = ROOT/relative
            assert path.stat().st_size == entry['bytes'] and sha(path) == entry['sha256'], relative
        print(f'Verified {len(inventory)} input files')
    if args.attempt:
        attempt = args.attempt
        manifest = read_json(attempt/'manifest.json')
        assert manifest['status'] == 'completed'
        assert manifest['completed_blocks'] == manifest['expected_blocks']
        source = read_json(HERE/'input-manifest.json')
        source_sha = sha(HERE/'input-manifest.json')
        assert manifest['input_manifest_sha256'] == source_sha
        if not manifest['smoke']:
            assert set(manifest['keys']) == {r['key'] for r in source['checkpoints']}
            assert manifest['blocks_per_checkpoint'] == 338 and manifest['device'] == 'cuda'
        summaries = []
        for key in manifest['keys']:
            artifact = read_json(attempt/(key+'.json.gz'))
            assert artifact['source'] == next(r for r in source['checkpoints'] if r['key'] == key)
            assert artifact['input_manifest_sha256'] == source_sha and artifact['grid'] == GRID
            coverage = artifact['coverage']
            assert coverage['complete_block_coverage'] == (not manifest['smoke'])
            assert coverage['sequences'] == manifest['blocks_per_checkpoint']
            assert coverage['input_tokens'] == manifest['blocks_per_checkpoint']*2048
            assert coverage['excluded_tail_tokens'] == (0 if manifest['smoke'] else 1444)
            assert coverage['full_source_tokens'] == 693668
            assert coverage['full_source_excluded_tail_tokens'] == 1444
            assert {r['name'] for r in artifact['rows']} == {f'{site}.layer_{layer}' for site in SITES for layer in range(6)}
            for row in artifact['rows']:
                validate_row(row)
                width = 512 if row['name'].startswith('h.layer_') else 128
                assert row['total'] == manifest['blocks_per_checkpoint']*2048*width
            for name, sites in GROUPS.items():
                assert artifact['groups'][name] == pool(artifact['rows'], name, sites)
            for site in SITES:
                assert artifact['sites'][site] == pool(artifact['rows'], site, (site,))
            if not manifest['smoke']:
                assert abs(artifact['reference_comparison']['loss_delta']) <= 5e-4
            summaries.append({'key':key, 'loss':coverage['loss'],
                              'seconds':artifact['evaluation_seconds'],
                              'peak_reserved_bytes':artifact['peak_reserved_bytes'],
                              'groups':{name:{field:group[field] for field in
                                             ('total','exact_zero_count','underflow','overflow','minimum','maximum')}
                                        for name,group in artifact['groups'].items()}})
        write_json(attempt/'verification.json', {'smoke':manifest['smoke'],
                   'blocks':manifest['completed_blocks'], 'checkpoints':summaries})
        inventory = {p.relative_to(attempt).as_posix():{'bytes':p.stat().st_size,'sha256':sha(p)}
                     for p in sorted(attempt.rglob('*')) if p.is_file() and p.name!='transfer-inventory.json'}
        write_json(attempt/'transfer-inventory.json', inventory)
        print(f'Verified {len(summaries)} checkpoints; smoke={manifest["smoke"]}; {len(inventory)} transfer files')
    if not args.inputs and args.attempt is None:
        parser.error('Use --inputs or --attempt')


if __name__ == '__main__':
    main()

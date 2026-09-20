"""Verify recovered coverage/selection and pool the recorded structural evidence."""
from collections import Counter

from io_utils import RUN, read, record, verify, write


def main():
    counts = Counter()
    final = []
    for path in sorted((RUN / 'artifacts/attempts').glob('*/result.json')):
        result = read(path)
        assert result['status'] == 'complete' and result['qualified']
        stage = path.parent.name.split('-')[0]
        counts[stage] += 1
        if stage == 'optimized':
            assert result['candidate'] == 'opt002' and result['validation_blocks'] == 338
            quality = read(path.with_name('quality.json'))
            assert all(quality['pass'].values())
            checks = quality['gates']['candidate_graph']
            assert len(checks) == 338 and all(c['pass'] and c['finite'] for c in checks)
            final.append({'condition': result['condition'], 'replicate': result['arguments']['replicate'],
                          'loss': quality['loss'], 'loss_delta': quality['loss_delta'],
                          'max_abs': max(c['max_abs'] for c in checks),
                          'max_relative_l2': max(c['relative_l2'] for c in checks),
                          'result': record(path), 'quality': record(path.with_name('quality.json'))})
    assert counts == {'smoke': 18, 'scientific': 54, 'development': 6, 'optimized': 6}
    assert {(x['condition'], x['replicate']) for x in final} == {(c, r) for c in ('c00', 'c21') for r in (1, 2, 3)}
    operator_counts = {}
    for name, expected in [('operator-checks', 28), ('control-checks', 72),
                           ('development/operator-opt002', 48), ('development/operator-opt003', 48)]:
        data = read(RUN / f'artifacts/{name}.json')
        assert data['status'] == 'passed' and len(data['checks']) == expected
        assert all(x.get('pass', True) for x in data['checks'])
        operator_counts[name] = expected
    for path in (RUN / 'candidates').glob('opt*/manifest.json'):
        for item in read(path)['files']:
            verify(item)
    selection = read(RUN / 'provenance/final-selection.json')
    assert selection['candidate'] == 'opt002'
    verify(selection['manifest'])
    for item in selection['development_evidence']:
        for key in ('result', 'timing'):
            verify(item[key])
    structural = []
    for condition in ('c00', 'c21'):
        path = RUN / f'artifacts/attempts/scientific-{condition}-full-r1-001/diagnostics.json'
        data = read(path)
        assert data['coverage']['blocks'] == 338
        sites = {}
        for site in ('a', 'm', 'h', 'z'):
            histograms = [v for k, v in data['active_features_per_row'].items() if k.startswith(site + '.')]
            rows = sum(sum(v) for v in histograms)
            short = sum(sum(v[:3]) for v in histograms)
            tiles = [v for k, v in data['empty_tile_counts'].items() if k.startswith(site + '.') and k.endswith(':8x16')]
            empty, total = sum(v['empty'] for v in tiles), sum(v['total'] for v in tiles)
            sites[site] = {'rows': rows, 'rows_at_most_two_nonzeros': short, 'short_row_fraction': short / rows,
                           'empty_8x16_tiles': empty, 'total_8x16_tiles': total, 'empty_8x16_fraction': empty / total}
        hybrid = dict(zip(data['hybrid_fields'], map(sum, zip(*data['hybrid_counts_by_layer'].values()))))
        structural.append({'condition': condition, 'sites': sites, 'hybrid_counts': hybrid, 'source': record(path)})
    write(RUN / 'results/closeout-verification.json', {
        'status': 'passed', 'attempt_counts': dict(counts), 'operator_checks': operator_counts,
        'final_quality': final, 'structural': structural, 'script': record(__file__),
        'limits': 'Rows/tile fractions pool integer counts over layers and all validation blocks. Short rows are occupancy counts; actual scalar/MMA counters remain separate.'})
    print(dict(counts), 'all qualified; candidate hashes and final selection verified')
    for row in structural:
        print(row['condition'], {site: round(v['short_row_fraction'] * 100, 5) for site, v in row['sites'].items()})


if __name__ == '__main__':
    main()

"""Reconcile retained evidence for the September 8 results package; no inference."""
from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from sparsity_research.ceilings import architecture_ceiling

SCALES = ('14M', '70M', '410M')
DOSES = (0., .01, .05, .1, .5)
CASE_DOSES = (0., .05, .5)
CASE_SITES = ('h','m','q_post','k_post','v')
FAMILIES = ('A0', 'A1-H', 'A1-H-L1', 'A1-H-OL1', 'A4', 'A4-OL1', 'A7', 'A7-OL1')
RENAMES = {'A1-H+L1': 'A1-H-L1', 'A1-H+OL1': 'A1-H-OL1',
           'A4+OL1@4': 'A4-OL1', 'A7+OL1@7': 'A7-OL1'}
OPS = ('qkv_projection', 'mlp_w1', 'mlp_w2', 'attention_output_projection', 'qk_scores', 'probability_value')
SITES = ('a', 'm', 'h', 'q_post', 'k_post', 'v', 'z', 'attention_output')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def one(rows, **fields):
    found = [r for r in rows if all(r.get(k) == v for k, v in fields.items())]
    if len(found) != 1:
        raise ValueError(f'Expected one {fields}, got {len(found)}')
    return found[0]


def close(a, b):
    if not math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-10):
        raise ValueError(f'Count/value disagreement: {a} != {b}')


def coverage(c):
    assert c['sequences'] == 338 and c['input_tokens'] == 692224
    assert c['excluded_tail_tokens'] == 1444 and c['complete_block_coverage']
    assert math.isfinite(c['loss'])


def logical_counts(c):
    for o in c['per_operation'].values():
        assert isinstance(o['product_count'], int) and isinstance(o['zero_product_count'], int)
        assert 0 <= o['zero_product_count'] <= o['product_count']
    assert sum(o['product_count'] for o in c['per_operation'].values()) == c['block_product_count']
    assert sum(o['zero_product_count'] for o in c['per_operation'].values()) == c['block_zero_product_count']
    assert c['model_product_count'] == c['block_product_count'] + c['lm_head_product_count']
    close(c['R_model'], c['block_zero_product_count'] / c['model_product_count'])


def mass_bands(row):
    n, z = row['total'], row['exact_zero_count']
    h1, h2 = row['threshold_hits']['0.001'], row['threshold_hits']['0.01']
    assert 0 <= z <= h1 <= h2 <= n
    return [z, h1-z, h2-h1, n-h2]


def frontier(rows):
    """Finite evaluated points only: minimize loss, maximize measured R."""
    return sorted([r for r in rows if not any(
        q['loss'] <= r['loss'] and q['R_model'] >= r['R_model'] and
        (q['loss'] < r['loss'] or q['R_model'] > r['R_model']) for q in rows)],
        key=lambda r: (r['R_model'], r['loss']))


def runtime_subset(raw):
    """Recompute the final retrospective on the 30 included checkpoints."""
    points = [dict(p, family=RENAMES.get(p['family'], p['family']))
              for p in raw['points'] if p['phase'] == 'final' and p['family'] != 'A4+OL1@h']
    selected = [p for p in points if p['candidate'] == 'k050']
    assert len(selected) == 30 and len({p['condition'] for p in selected}) == 30
    assert all(p['qualified'] and len(p['replicates']) == 3 for p in selected)
    summary = {}
    for candidate in sorted({p['candidate'] for p in points}):
        values = [p['speedup'] for p in points if p['candidate'] == candidate and p['qualified']]
        summary[candidate] = {'n': len(values), 'geomean': math.exp(sum(map(math.log, values))/len(values)),
                              'minimum': min(values), 'maximum': max(values),
                              'above_one': sum(v > 1 for v in values)}
    x = [p['R_model'] for p in selected]; y = [p['speedup'] for p in selected]
    mx, my = sum(x)/len(x), sum(y)/len(y)
    slope = sum((a-mx)*(b-my) for a, b in zip(x,y))/sum((a-mx)**2 for a in x)
    intercept = my-slope*mx
    r2 = 1-sum((b-intercept-slope*a)**2 for a,b in zip(x,y))/sum((b-my)**2 for b in y)
    ablations = {}
    for candidate in ('k050-no-skip', 'k050-attention-dense'):
        ratios = [p['speedup']/q['speedup'] for p in selected for q in points
                  if q['candidate'] == candidate and q['condition'] == p['condition'] and q['qualified']]
        ablations[candidate] = {'n': len(ratios), 'geomean_ratio': math.exp(sum(map(math.log, ratios))/len(ratios)),
                                'k050_faster': sum(v > 1 for v in ratios)}
    # Progress was measured on c01/c11/c25/c30, all inside this cohort.
    assert all(int(p['condition'][1:]) <= 30 for p in raw['points'] if p['phase'] == 'history')
    return {'points': points, 'progress': raw['progress'], 'final_candidates': summary,
            'k050_regression': {'n': len(x), 'slope_per_fraction': slope, 'intercept': intercept, 'r2': r2},
            'ablations': ablations, 'physical_gpu_uuid': raw['physical_gpu_uuid'], 'metric': raw['metric'],
            'timing_block_indices': raw['timing_block_indices']}


def load_evidence():
    sources = {}

    def record(path):
        path = Path(path)
        sources[path.relative_to(ROOT).as_posix()] = sha(path)
        return path

    def source(path):
        return read(record(path))

    prior = source(next((ROOT/'analyses').glob('013-*/figure_data.json')))
    # Every explicit source in the retained synthesis must still exist and match.
    for path, digest in prior['sources'].items():
        assert sha(ROOT/path) == digest, path
    record(ROOT/'manuscript/draft/main.pdf')
    record(ROOT/'manuscript/artifacts/pythia-architecture-sparsification-ladder.pdf')
    for name in ('main', 'introduction', 'methodology', 'experimental-study', 'experimental-appendix'):
        record(ROOT/f'manuscript/draft/{name}.tex')
    record(ROOT/'src/sparsity_research/ceilings.py')

    control_paths = {}
    for scale, run in (('70M', '018'), ('410M', '019')):
        for path in (ROOT/'runs').glob(f'{run}-*/artifacts/attempts/*/manifest.json'):
            m = read(path)
            for cid, family in (('a0-gelu', 'A0'), ('a1h-relu', 'A1-H')):
                if m['condition']['id'] == cid and m['status'] == 'completed':
                    assert (scale, family) not in control_paths
                    control_paths[scale, family] = path.parent

    trained = []
    for old in prior['trained']:
        if old['family'] == 'A4+OL1@h':
            continue
        scale, family = old['scale'], RENAMES.get(old['family'], old['family'])
        folder = ROOT/old['source'] if 'source' in old else control_paths[scale, family]
        m = source(folder/'manifest.json')
        cfg = yaml.safe_load(record(folder/'config.yaml').read_text(encoding='utf-8-sig'))
        logical = source(folder/'diagnostics/logical_products.json')
        act = source(folder/'diagnostics/activation_statistics.json')
        coverage(logical['coverage'])
        c = logical['measured']
        logical_counts(c)
        if 'source' in old:
            close(old['R_model'], c['R_model'])
            close(old['loss'], logical['coverage']['loss'])
        assert m['status'] == 'completed' and m['completed_steps'] == 712
        assert m['input_tokens'] == 1493172224 and m['seeds'] == {'model': 1234, 'data_order': 1234}
        assert not m['model']['loaded_checkpoint_weights']
        arch = logical['architecture_maximum']
        calc = architecture_ceiling(arch['topology_id'], **{k: arch[k] for k in
            ('layers', 'hidden_size', 'ffn_size', 'sequence_length', 'vocabulary_size')})
        for key in ('reachable_product_count', 'model_product_count', 'block_product_count'):
            assert calc[key] == arch[key]
        assert c['model_product_count'] == 338 * calc['model_product_count']
        site_data = {}
        for pooled in act['pooled_by_site']:
            layers = [r for r in act['rows'] if r['name'].split('.layer_')[0] == pooled['name']]
            assert len(layers) == arch['layers']
            for key in ('total', 'finite', 'nonfinite', 'exact_zero_count'):
                assert sum(r[key] for r in layers) == pooled[key]
            assert pooled['nonfinite'] == 0
            width = arch['ffn_size'] if pooled['name'] == 'h' else arch['hidden_size']
            assert pooled['total'] == 692224 * arch['layers'] * width
            for eps, hits in pooled['threshold_hits'].items():
                assert sum(r['threshold_hits'][eps] for r in layers) == hits
            close(pooled['rms'], math.sqrt(sum(r['square_sum'] for r in layers)/pooled['finite']))
            site_data[pooled['name']] = dict(pooled, mass_bands=mass_bands(pooled))
        reach = set(calc['reachable_operations'])
        inside = sum(c['per_operation'][op]['zero_product_count'] for op in reach)
        outside = c['block_zero_product_count'] - inside
        nreach = calc['reachable_product_count'] * 338
        trained.append({
            'id': f'{scale}:{family}:{old["dose"]}', 'scale': scale, 'family': family,
            'dose': old['dose'], 'kind': 'trained', 'loss': logical['coverage']['loss'],
            'terminal_loss': m['validation_coverage']['loss'], 'R_model': c['R_model'],
            'counts': c, 'ceiling': calc, 'sites': site_data,
            'U_arch': c['block_zero_product_count']/nreach if nreach else None,
            'U_reach': inside/nreach if nreach else None,
            'outside_reach_zero_products': outside,
            'source': folder.relative_to(ROOT).as_posix(),
            'identity': {'initial_parameter_sha256': m['initial_parameter_sha256'],
                         'schedule_sha256': m['training_schedule_hash'],
                         'validation_sha256': m['data']['validation']['tokens_sha256'],
                         'training_tokens': m['input_tokens'], 'seeds': m['seeds'],
                         'training': cfg['training'], 'pressure': m['activation_pressure']},
            })
    for scale in SCALES:
        group = [r for r in trained if r['scale'] == scale]
        for key in ('initial_parameter_sha256', 'schedule_sha256', 'validation_sha256'):
            assert len({r['identity'][key] for r in group}) == 1, (scale, key)

    # Use raw evaluated p=0 rows, never display anchors from older frontier plots.
    a6path = next((ROOT/'analyses').glob('006-*/teal_all_variants.json'))
    a6 = source(a6path)
    assert set(a6['method']['sites']) == {'a','m','h','z'}
    clipping = []
    for row in a6['conditions']:
        assert {name.split('.layer_')[0] for name in row['thresholds_by_site_layer']} == {'a','m','h','z'}
        coverage(row['validation'])
        c = row['logical_products']
        logical_counts(c)
        clipping.append({'id': f'14M:clip:{row["condition_id"]}:{row["target_sparsity"]}',
                         'scale': '14M', 'family': row['condition_id'], 'kind': 'clipped',
                         'dose': row['target_sparsity'], 'loss': row['validation']['loss'],
                         'R_model': c['R_model'], 'counts': c,
                         'source': a6path.relative_to(ROOT).as_posix()})
    assert len(clipping) == 150
    a11 = source(next((ROOT/'analyses').glob('011-*/figure_data.json')))
    large_clipping = {}
    for scale, run in (('70M', '018'), ('410M', '019')):
        path = next((ROOT/'runs').glob(f'{run}-*/artifacts/teal/teal_frontiers.json'))
        assert sha(path) == a11['sources'][path.relative_to(ROOT).as_posix()]
        large_clipping[scale] = (path, source(path))
    for row in a11['teal_points']:
        if row['scale'] == '14M':
            continue
        path, raw = large_clipping[row['scale']]
        cid = 'a0-gelu' if row['control'] == 'A0' else 'a1h-relu'
        measured = one(raw['points'], condition_id=cid, target_sparsity=row['target_sparsity'])
        assert {name.split('.layer_')[0] for name in measured['thresholds_by_site_layer']} == {'a','m','h','z'}
        coverage(measured['validation'])
        c = measured['logical_products']
        logical_counts(c)
        close(row['R_model'], c['block_zero_product_count']/c['model_product_count'])
        close(row['validation_loss'], measured['validation']['loss'])
        clipping.append({'id': f'{row["scale"]}:clip:{row["control"]}:{row["target_sparsity"]}',
                         'scale': row['scale'], 'family': row['control'], 'kind': 'clipped',
                         'dose': row['target_sparsity'], 'loss': row['validation_loss'],
                         'R_model': row['R_model'], 'counts': c,
                         'source': path.relative_to(ROOT).as_posix()})
    for scale in SCALES:
        groups = {r['family'] for r in clipping if r['scale'] == scale}
        for f in groups:
            assert sorted(r['dose'] for r in clipping if r['scale'] == scale and r['family'] == f) == [i/10 for i in range(10)]

    ceilings = []
    for scale in SCALES:
        arch = one(trained, scale=scale, family='A0')['ceiling']
        kwargs = {k: arch[k] for k in ('layers','hidden_size','ffn_size','sequence_length','vocabulary_size')}
        for family, topology in (('A0','A0'), ('A1-H','A1-H'), ('A4','A4-Z'), ('A7','A7-Z-POST')):
            ceilings.append({'scale': scale, 'family': family,
                             'parameters': one(prior['exposure'], scale=scale)['parameters'],
                             'unit': 'logical scalar products per one uncached 2048-token sequence',
                             **architecture_ceiling(topology, **kwargs)})
        clip_ceiling = one(ceilings, scale=scale, family='A4')
        baseline = one(trained, scale=scale, family='A0')['loss']
        for row in [r for r in clipping if r['scale'] == scale]:
            row['ceiling'] = clip_ceiling
            row['normalization_sites'] = ['a','m','h','z']
            assert row['counts']['model_product_count'] == 338*clip_ceiling['model_product_count']
            row['U_arch'] = row['counts']['block_zero_product_count']/(338*clip_ceiling['reachable_product_count'])
            row['delta_loss_vs_A0'] = row['loss']-baseline
            row['control'] = {'gelu-control': 'A0', 'relu-control': 'A1-H', 'A0': 'A0', 'A1-H': 'A1-H'}.get(row['family'])

    contrasts = []

    def contrast(block, reference, treatment, level, level_kind):
        contrasts.append({'block': block, 'reference': reference['id'], 'treatment': treatment['id'],
                          'dose': level, 'dose_kind': level_kind,
                          'delta_loss': treatment['loss']-reference['loss'],
                          'delta_R_pp': 100*(treatment['R_model']-reference['R_model'])})

    small = [r for r in trained if r['scale'] == '14M']
    get = lambda f, k=None: one(small, family=f, dose=k)
    contrast('GELU to ReLU', get('A0'), get('A1-H'), None, 'none')
    for lam in (.05, .1, .5, 1.):
        contrast('Add L1 at h', get('A1-H'), get('A1-H-L1', lam), lam, 'lambda')
        contrast('L1 to OL1 at h', get('A1-H-L1', lam), get('A1-H-OL1', lam), lam, 'lambda')
    for k in DOSES:
        contrast('A1-H to A4', get('A1-H'), get('A4', k), k, 'kappa')
        for label, p, q in (('Add OL1 to A4', 'A4', 'A4-OL1'),
                            ('A4 to A7 gates', 'A4', 'A7'), ('Add OL1 to A7', 'A7', 'A7-OL1')):
            contrast(label, get(p, k), get(q, k), k, 'kappa')
    scale_pairs = []
    for scale in SCALES:
        base = one(trained, scale=scale, family='A0')
        for r in [r for r in trained if r['scale'] == scale]:
            r['delta_loss_vs_A0'] = r['loss'] - base['loss']
        for k in DOSES:
            a = one(trained, scale=scale, family='A4-OL1', dose=k)
            b = one(trained, scale=scale, family='A7-OL1', dose=k)
            scale_pairs.append({'scale': scale, 'dose': k, 'delta_loss': b['loss']-a['loss'],
                                'delta_R_pp': 100*(b['R_model']-a['R_model']),
                                'delta_U_pp': 100*(b['U_arch']-a['U_arch'])})
    fsets = {'14M_main_trained': small,
             '14M_all_trained_and_clipped': small+[r for r in clipping if r['scale']=='14M']}
    frontiers = {name: [r['id'] for r in frontier(rows)] for name, rows in fsets.items()}
    # Figure 01 follows the evaluated dose sweeps; Pareto selection is tabulated separately.
    overview_series = {family: [r['id'] for r in sorted(
        [r for r in small if r['family']==family], key=lambda r: r['dose'] or 0)]
        for family in FAMILIES}
    for family in ('A0','A1-H'):
        overview_series[family+' + clipping'] = [r['id'] for r in sorted(
            [r for r in clipping if r['scale']=='14M' and r['control']==family],
            key=lambda r: r['dose'])]
    for source_family in sorted({r['family'] for r in clipping
                                 if r['scale']=='14M' and r['control'] is None}):
        overview_series[source_family+' + clipping'] = [r['id'] for r in sorted(
            [r for r in clipping if r['scale']=='14M' and r['family']==source_family],
            key=lambda r:r['dose'])]
    runtime = runtime_subset(source(next((ROOT/'runs').glob('029-*/results/matched-retrospective-001.json'))))
    for point in runtime['points']:
        matched = [r for r in small if r['family'] == point['family']
                   and math.isclose(r['R_model'],point['R_model'],rel_tol=1e-10,abs_tol=1e-10)]
        assert len(matched) == 1
        point['evidence_id'] = matched[0]['id']
    return {'trained': trained, 'clipping': clipping, 'contrasts': contrasts, 'scale_pairs': scale_pairs,
            'frontiers': frontiers, 'overview_series': overview_series,
            'sources': sources, 'exposure': prior['exposure'],
            'runtime': runtime, 'ceilings': ceilings,
            'activation_case': {'kappas': list(CASE_DOSES), 'sites': list(CASE_SITES),
                                'bin_status': 'four retained bins; finer measurement pending design confirmation'},
            'coverage': {'documents': 500, 'sequences': 338, 'input_tokens': 692224,
                         'prediction_tokens': 691886, 'excluded_tail_tokens': 1444, 'seed_count': 1},
            'normalization': 'U_arch = all observed zero products / selected-site reachable products; not bounded by one. U_reach excludes outside-reach zero products.',
            'activation_bands': ['exact zero', '0 < abs(x) <= 0.001', '0.001 < abs(x) <= 0.01', 'abs(x) > 0.01']}

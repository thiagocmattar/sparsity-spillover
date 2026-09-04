"""Reduce completed experiments for one manuscript; no model execution."""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SITES = ('a', 'm', 'h', 'q_post', 'k_post', 'v', 'z')
OPS = ('qkv_projection', 'mlp_w1', 'mlp_w2', 'attention_output_projection',
       'qk_scores', 'probability_value')
DOSES = (0., .01, .05, .1, .5)
FAMILIES = {'gelu_control': 'A0', 'relu_control': 'A1-H',
            'a1h_naive_l1': 'A1-H+L1', 'a1h_ol1': 'A1-H+OL1',
            'a4z_threshold': 'A4', 'a4z_ol1': 'A4+OL1@4',
            'a7z_post_mixed_threshold': 'A7',
            'a7z_post_mixed_threshold_ol1': 'A7+OL1@7'}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def one(rows, **keys):
    matches = [r for r in rows if all(r.get(k) == v for k, v in keys.items())]
    if len(matches) != 1:
        raise ValueError(f'Expected one row for {keys}, got {len(matches)}')
    return matches[0]


def close(a, b, tol=1e-10):
    if not math.isclose(a, b, rel_tol=tol, abs_tol=tol):
        raise ValueError(f'Inconsistent values: {a} versus {b}')


def paired_delta(parent, child):
    if parent['scale'] != child['scale'] or parent['dose'] != child['dose']:
        raise ValueError('Paired contrast must match scale and dose')
    return {'delta_loss': child['loss'] - parent['loss'],
            'delta_R_pp': 100 * (child['R_model'] - parent['R_model'])}


def nondominated(rows):
    return [r for r in rows if not any(
        q['loss'] <= r['loss'] and q['R_model'] >= r['R_model'] and
        (q['loss'] < r['loss'] or q['R_model'] > r['R_model']) for q in rows)]


def load_evidence():
    hashes = {}

    def source(path):
        path = Path(path)
        hashes[path.relative_to(ROOT).as_posix()] = digest(path)
        return read(path)

    a8 = source(next((ROOT / 'analyses').glob('008-*/figure_data.json')))
    a9 = source(next((ROOT / 'analyses').glob('009-*/figure_data.json')))
    a11 = source(next((ROOT / 'analyses').glob('011-*/figure_data.json')))
    a12 = source(next((ROOT / 'analyses').glob('012-*/figure_data.json')))
    lr_screen = source(next((ROOT / 'runs').glob('021-*/artifacts/selection.json')))

    # Validate hashes recorded by previous reductions wherever paths are explicit.
    def check_refs(obj):
        if isinstance(obj, dict):
            if 'path' in obj and 'sha256' in obj:
                p = ROOT / obj['path']
                if p.is_file() and digest(p) != obj['sha256']:
                    raise ValueError(f'Upstream identity mismatch: {p}')
            for k, v in obj.items():
                if isinstance(v, str) and len(v) == 64 and '/' in k:
                    p = ROOT / k
                    if p.is_file() and digest(p) != v:
                        raise ValueError(f'Upstream identity mismatch: {p}')
                if isinstance(v, str) and k.endswith('_sha256'):
                    path_key = k.removesuffix('_sha256')
                    if path_key in obj and isinstance(obj[path_key], str):
                        p = ROOT / obj[path_key]
                        if p.is_file() and digest(p) != v:
                            raise ValueError(f'Upstream identity mismatch: {p}')
                check_refs(v)
        elif isinstance(obj, list):
            for v in obj:
                check_refs(v)
    for prior in (a8, a9, a11, a12):
        check_refs(prior)

    def trained(run, attempt, family, dose, scale, terminal, expected_r):
        folder = one_path(list((ROOT / 'runs').glob(f'{run}-*/artifacts/attempts/{attempt}')))
        logical = source(folder / 'diagnostics/logical_products.json')
        act = source(folder / 'diagnostics/activation_statistics.json')
        cov, counts = logical['coverage'], logical['measured']
        assert cov['sequences'] == 338 and cov['input_tokens'] == 692224
        assert cov['excluded_tail_tokens'] == 1444 and cov['complete_block_coverage']
        assert sum(o['product_count'] for o in counts['per_operation'].values()) == counts['block_product_count']
        assert sum(o['zero_product_count'] for o in counts['per_operation'].values()) == counts['block_zero_product_count']
        assert counts['block_product_count'] + counts['lm_head_product_count'] == counts['model_product_count']
        r = counts['block_zero_product_count'] / counts['model_product_count']
        close(r, expected_r)
        close(r, counts['R_model'])
        sites = {}
        for item in act['pooled_by_site']:
            close(item['exact_zero_fraction'], item['exact_zero_count'] / item['total'])
            sites[item['name']] = {'zeros': item['exact_zero_count'], 'count': item['total'],
                                   'fraction': item['exact_zero_count'] / item['total'],
                                   'rms': item['rms']}
        return {'scale': scale, 'family': family, 'dose': dose, 'kind': 'trained',
                'loss': cov['loss'], 'terminal_loss': terminal, 'R_model': r,
                'R_block': counts['block_zero_product_count'] / counts['block_product_count'],
                'counts': counts, 'architecture': logical['architecture_maximum'],
                'sites': sites, 'source': folder.relative_to(ROOT).as_posix()}

    rows = []
    for r in a8['trained_endpoints']:
        rows.append(trained(r['run'].removeprefix('run'), r['attempt_id'],
                            FAMILIES[r['series_id']], r['dose'], '14M',
                            r['final_validation_loss'], r['R_model']))
    for r in a9['series']:
        if r['series_id'] == 'run012_a4_ol1_h_only' or r['run'].startswith('012-'):
            rows.append(trained('012', r['attempt_id'], 'A4+OL1@h', r['kappa'],
                                '14M', r['final_validation_loss'], r['R_model']))
    for r in a11['trained_endpoints']:
        if r['scale'] != '14M':
            path = next(p for p in r['source_files'] if p.endswith('/diagnostics/logical_products.json'))
            run = Path(path).parts[1][:3]
            fam = {'A4-OL1': 'A4+OL1@4', 'A7-OL1': 'A7+OL1@7'}[r['family']]
            rows.append(trained(run, r['attempt_id'], fam, r['kappa'], r['scale'],
                                r['execution_validation_loss'], r['R_model']))

    clipping = []
    for r in a11['teal_points']:
        close(r['R_model'], r['logical_counts']['zero_product_count'] / r['logical_counts']['model_product_count'])
        clipping.append({'scale': r['scale'], 'family': r['control'] + ' clipped',
                         'kind': 'clipped', 'dose': r['target_sparsity'],
                         'loss': r['validation_loss'], 'R_model': r['R_model']})
    # Controls at larger scales have their paired zero-target evaluation already retained.
    for scale in ('70M', '410M'):
        for fam in ('A0', 'A1-H'):
            r = dict(one(clipping, scale=scale, family=fam + ' clipped', dose=0.))
            r.update(family=fam, kind='trained', dose=None)
            rows.append(r)
    assert len([r for r in rows if r['scale'] == '14M']) == 35
    assert len(rows) == 59 and len(clipping) == 60

    contrasts = []
    for k in DOSES:
        get = lambda f: one(rows, scale='14M', family=f, dose=k)
        for parent, child in (('A4', 'A7'), ('A4', 'A4+OL1@4'), ('A7', 'A7+OL1@7'),
                              ('A4', 'A4+OL1@h'), ('A4+OL1@h', 'A4+OL1@4')):
            contrasts.append({'parent': parent, 'child': child, 'dose': k,
                              **paired_delta(get(parent), get(child))})
    differences = []
    for k in DOSES:
        a = one(contrasts, parent='A4', child='A4+OL1@4', dose=k)
        b = one(contrasts, parent='A7', child='A7+OL1@7', dose=k)
        differences.append({'dose': k, 'delta_loss': b['delta_loss']-a['delta_loss'],
                            'delta_R_pp': b['delta_R_pp']-a['delta_R_pp']})
    runtime_path = next((ROOT / 'runs').glob('023-*/results/sentinel-summary.csv'))
    hashes[runtime_path.relative_to(ROOT).as_posix()] = digest(runtime_path)
    runtime = list(csv.DictReader(runtime_path.open(encoding='utf-8-sig', newline='')))
    verification = source(runtime_path.parent / 'raw/verification.json')
    for r in runtime:
        for k, v in r.items():
            if k != 'condition_id' and v:
                r[k] = float(v)
    assert len(runtime) == 6
    return {'sources': hashes, 'trained': rows, 'clipping': clipping,
            'contrasts': contrasts, 'pressure_response_differences': differences,
            'exposure': a12['baseline_exposure'], 'runtime': runtime,
            'runtime_verification': verification,
            'lr_screen': lr_screen,
            'coverage': {'sequences': 338, 'input_tokens': 692224, 'excluded_tail_tokens': 1444},
            'loss_policy': 'same eager pass as logical products; terminal loss retained separately',
            'inference_scope': 'one seed; complete validation; full-sequence T=2048'}


def one_path(paths):
    if len(paths) != 1:
        raise ValueError(f'Ambiguous source paths: {paths}')
    return paths[0]

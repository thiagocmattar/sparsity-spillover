"""Audit retained 70M measurements; no model execution or timing experiments."""
import hashlib
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RUN = ROOT / 'runs/045-2026-09-20-pythia70m-kernel-grid'
SOURCES = {}


def read(path, expected=None):
    path = Path(path)
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if expected is not None:
        assert digest == expected, path
    SOURCES[path.relative_to(ROOT).as_posix()] = digest
    return json.loads(data)


def structure(path):
    d = read(path)
    assert d['status'] == 'complete' and d['coverage']['blocks'] == 338
    assert d['hybrid_fields'] == [
        'h_issued', 'h_bypassed', 'z_issued', 'z_bypassed',
        'h_scalar_products', 'z_scalar_products']
    work = [sum(v[i] for v in d['hybrid_counts_by_layer'].values()) for i in range(6)]
    output = {}
    for site, width, offset in [('h', 2048, 0), ('z', 512, 2)]:
        layers = [d['active_features_per_row'][f'{site}.layer_{i}'] for i in range(6)]
        assert all(len(v) == width + 1 for v in layers)
        hist = [sum(v[i] for v in layers) for i in range(width + 1)]
        rows, active = sum(hist), sum(i * n for i, n in enumerate(hist))
        pooled = next(v for v in d['pooled_by_site'] if v['name'] == site)
        assert rows == 338 * 2048 * 6
        assert rows * width == pooled['total']
        assert active == pooled['total'] - pooled['exact_zero_count']
        tiles = [d['empty_tile_counts'][f'{site}.layer_{i}:8x16'] for i in range(6)]
        issued, bypassed = work[offset:offset + 2]
        output[site] = {
            'rows': rows, 'exact_zero_count': pooled['exact_zero_count'],
            'elements': pooled['total'],
            'exact_zero_percent': 100 * pooled['exact_zero_count'] / pooled['total'],
            'active_elements': active, 'mean_nnz_per_row': active / rows,
            'rows_nnz_le8': sum(hist[:9]), 'rows_nnz_le8_percent': 100 * sum(hist[:9]) / rows,
            'empty_8x16_tiles': sum(v['empty'] for v in tiles),
            'total_8x16_tiles': sum(v['total'] for v in tiles),
            'empty_8x16_percent': 100 * sum(v['empty'] for v in tiles) / sum(v['total'] for v in tiles),
            'mma_issued': issued, 'mma_bypassed': bypassed,
            'mma_issued_percent': 100 * issued / (issued + bypassed),
            'scalar_products_including_duplicates': work[4 + offset // 2],
            'per_layer': [
                {'layer': i, 'rows': sum(v), 'rows_nnz_le8': sum(v[:9]),
                 'mean_nnz_per_row': sum(k * n for k, n in enumerate(v)) / sum(v)}
                for i, v in enumerate(layers)],
        }
    return output


def profile(path):
    d = read(path)
    r = next(r for r in d['profiles'] if r['execution'] == 'graph' and r['mode'] == 'candidate')
    assert r['inputs'] == 4
    groups = {}
    # Native GEMMs in this candidate are a/m; its full head is the distinct CUTLASS kernel.
    for name, value in r['kernels'].items():
        if 'void joint<' in name:
            group = 'hz_fallback'
        elif 'parallel_rows<' in name:
            group = 'hz_prepass'
        elif '_ZN7cutlass6Kernel' in name:
            group = 'dense_head'
        elif 'run035_flash' in name:
            group = 'dense_attention'
        elif 'Kernel2' in name:
            group = 'native_am'
        else:
            group = 'other'
        groups[group] = groups.get(group, 0) + value['duration_us'] / r['inputs'] / 1000
    assert all(k in groups for k in ('hz_fallback', 'hz_prepass', 'dense_head', 'dense_attention', 'native_am'))
    return groups


def main():
    grid = read(RUN / 'results/matched-grid.json')
    figures = read(ROOT / 'analyses/034-2026-09-21-70m-latency-quality-frontier/data/figure-data.json')
    canonical = {p['run045_id']: p for p in figures['points'] if p.get('run045_id')}
    assert grid['checkpoints'] == 26 and grid['fresh_processes'] == 78 and grid['all_qualified']
    rows = []
    processes = 0
    for r in grid['conditions']:
        values = {k: [] for k in r['latency_ms']}
        replicate_one = None
        for source in r['source_records']:
            path = RUN / source['path']
            data = read(path, source['sha256'])
            assert path.stat().st_size == source['bytes']
            if path.name != 'result.json':
                continue
            processes += 1
            assert data['status'] == 'complete' and data['qualified'] and data['validation_blocks'] == 338
            timing = read(path.with_name('timing.json'))
            assert len(timing['indices']) == len(set(timing['indices'])) == 64
            assert len(timing['samples']) == 3 * 64 * 7
            for sample in timing['samples']:
                assert sample['output_shape'] == [1, 2048, 50304]
                values[sample['mode']].append(sample['host_ms'])
            if data['arguments']['replicate'] == 1:
                assert replicate_one is None
                replicate_one = path.parent
        for mode, samples in values.items():
            assert len(samples) == 1344 and all(v > 0 for v in samples)
            measured = math.exp(math.fsum(map(math.log, samples)) / len(samples))
            assert math.isclose(measured, r['latency_ms'][mode], rel_tol=1e-12)
        entry = {'id': r['id'], 'family': r['family'], 'kappa': r['dose'],
                 'latency_ms': r['latency_ms'], 'native_base_speedup': r['native_base_speedup'],
                 'operand_structure_bf16': structure(replicate_one / 'diagnostics-candidate.json')}
        if r['id'] in canonical:
            p = canonical[r['id']]
            assert p['qualified'] == r['qualified']
            assert p['implementation_latency_ms'] == r['latency_ms']
            assert math.isclose(p['sparsity'], 100 * p['zero_product_count'] / p['model_product_count'], rel_tol=1e-12)
            entry.update(canonical_loss=p['loss'], canonical_R_model_percent=p['sparsity'],
                         canonical_zero_product_count=p['zero_product_count'],
                         canonical_model_product_count=p['model_product_count'])
        if (replicate_one / 'profile-summary.json').exists():
            entry['instrumented_gpu_ms_per_input'] = profile(replicate_one / 'profile-summary.json')
        rows.append(entry)
    assert processes == 78
    controls = read(ROOT / 'analyses/028-2026-09-20-70m-optimized-grid/data/retained-controls.json')
    for path, digest in controls['sources_sha256'].items():
        read(ROOT / path, digest)
    selection = read(ROOT / 'runs/042-2026-09-20-pythia70m-sparse-scale/provenance/final-selection.json')
    result = {
        'scope': 'CPU reduction of retained evidence only; no new GPU qualification or timing.',
        'verification': {'qualified_process_records_checked': processes,
                         'checkpoint_latencies_recomputed': len(rows),
                         'full_validation_structural_diagnostics_checked': len(rows)},
        'native_base_ms_run045': grid['native_base_ms'], 'conditions': rows,
        'run042_controls_separate_session': controls,
        'run042_selection_criterion': selection['criterion'],
        'interpretation': [
            'Canonical FP16 quality/logical opportunity and BF16 operand diagnostics are separate.',
            'NNZ<=8 is a necessary occupancy condition, not proof of fast-path eligibility or joint group completion.',
            'MMA issued fractions use this implementation\'s padded instruction denominator, not runtime or R_model.',
            'Profile values are summed instrumented GPU times over four inputs, not benchmark host latency.',
            'Conditional replacement effects and different sessions must not be added or pooled.'],
        'sources_sha256': SOURCES,
    }
    dest = HERE / 'data/assessment.json'
    dest.parent.mkdir(exist_ok=True)
    dest.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({**result['verification'], 'hashed_sources': len(SOURCES),
                      'output': dest.relative_to(ROOT).as_posix()}))


if __name__ == '__main__':
    main()

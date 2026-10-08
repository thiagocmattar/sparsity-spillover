"""Combine completed Runs 064/065 without changing their measurements."""
import collections
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
RUN14 = REPO / 'runs/064-2026-10-07-pythia14m-forward-latency-decomposition'
RUNBIG = REPO / 'runs/065-2026-10-07-larger-model-forward-decomposition'

CATEGORIES = {
    'positions': 'Position indices', 'embedding': 'Token embedding',
    'rotary_tables': 'RoPE tables', 'norm_attention': 'Block layer norms',
    'norm_mlp': 'Block layer norms', 'norm_pair': 'Block layer norms',
    'qkv': 'QKV projection', 'rope_qkv_layout': 'RoPE / QKV layout',
    'attention': 'Attention', 'context_pack': 'Context layout',
    'mlp_up': 'MLP up projection', 'h_gate': 'H gate', 'z_gate': 'Z gate',
    'attention_out': 'Z down projection (gate fused if applicable)',
    'z_gate_dense_down': 'Z down projection (gate fused if applicable)',
    'mlp_down': 'H down projection', 'h_dense_down': 'H down projection',
    'h_gathered_down': 'H down projection',
    'h_feature_compaction': 'H feature compaction',
    'joint_gate_down_residual': 'Joint H/Z gates / down / residual',
    'residual': 'Residual sum', 'final_norm': 'Final norm',
    'lm_head': 'Vocabulary projection',
}
ORDER = list(dict.fromkeys(CATEGORIES.values()))
BRANCH = {'H gate', 'Z gate', 'Z down projection (gate fused if applicable)',
          'H down projection', 'H feature compaction',
          'Joint H/Z gates / down / residual', 'Residual sum'}
LABELS = {
    'native': 'Native dense arithmetic',
    'generic_sparse': 'Generic sparse consumer',
    'generic_no_skip': 'Generic consumer, skipping disabled',
    'tile_sparse': 'Corner-specialized sparse consumer',
    'tile_no_skip': 'Corner-specialized consumer, skipping disabled',
    'sparse': 'Existing scalar-threshold sparse consumer',
    'no_skip': 'Existing scalar consumer, skipping disabled',
}


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((json.dumps(value, indent=2, allow_nan=False) + '\n').encode())


def record(path):
    return {'path': path.relative_to(REPO).as_posix(),
            'bytes': path.stat().st_size,
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def csvout(name, rows):
    with (ROOT / 'results' / name).open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def collect():
    d14 = read(RUN14 / 'results/decomposition.json')
    dbig = read(RUNBIG / 'results/decomposition.json')
    assert d14['status'] == dbig['status'] == 'verified'
    points = []
    for pair, condition in d14['pairs'].items():
        points.append(('14M', pair, condition, RUN14, 'tile_sparse', 'tile_no_skip'))
    for condition in dbig['conditions'].values():
        points.append((condition['size'].upper(), condition['pair'], condition,
                       RUNBIG, 'sparse', 'no_skip'))
    output = {'status': 'verified', 'category_order': ORDER,
              'conditions': [], 'measurement_boundary':
              'Resident-input, synchronized host CUDA-graph replay; B1/T2048, BF16, full 50304-way logits.'}
    timing_rows, stage_rows, category_rows, layer_rows, node_rows = [], [], [], [], []
    for size, pair, condition, source, selected, no_skip in points:
        cfg = read(source / 'config.json')
        assert [cfg[k] for k in ['timing_inputs', 'timing_passes', 'process_replicates',
                                'validation_blocks', 'validation_documents',
                                'excluded_tail_tokens', 'trace_inputs', 'trace_replays']] == [64, 7, 3, 338, 500, 1444, 4, 3]
        assert len(condition['processes']) == 3
        uuids = {p['gpu_uuid'] for p in condition['processes']}
        assert len(uuids) == 1
        key = f'{size}-k{pair}'
        modes = {}
        for mode, data in condition['modes'].items():
            host = data['baseline_host_us']
            trace = data['trace_budget']
            assert host['count'] == 1344
            assert trace['all_signatures_match'] and len(trace['replays']) == 12
            assert abs(sum(s['mean'] for s in trace['stages']) - trace['mean_active_us']) < .1
            cats = collections.defaultdict(float)
            layers = collections.defaultdict(float)
            events = {s['stage']: s['mean_us'] for s in data['event_partition']['stages']}
            for stage in trace['stages']:
                name = stage['stage']
                category = CATEGORIES[name.split('.')[-1]]
                cats[category] += stage['mean']
                if name.startswith('layer') and category in BRANCH:
                    layers[int(name.split('.')[0][5:])] += stage['mean']
                stage_rows.append({'condition': key, 'model': size, 'kappa': pair.replace('0p1', '0.1'),
                    'tiled_threshold': size == '14M', 'mode': mode, 'stage': name,
                    'category': category, 'cupti_mean_us': stage['mean'],
                    'cupti_p10_us': stage['p10'], 'cupti_p90_us': stage['p90'],
                    'active_share': stage['mean'] / trace['mean_active_us'],
                    'event_mean_us': events[name]})
            assert set(layers) == set(range(6))
            for label, value in cats.items():
                category_rows.append({'condition': key, 'model': size, 'mode': mode,
                                     'category': label, 'cupti_mean_us': value,
                                     'active_share': value / trace['mean_active_us']})
            for layer, value in layers.items():
                layer_rows.append({'condition': key, 'model': size, 'mode': mode,
                                   'layer': layer, 'branch_bundle_cupti_us': value})
            for node in trace['kernel_nodes']:
                node_rows.append({'condition': key, 'model': size, 'mode': mode,
                                 'index': node['index'], 'stage': node['stage'],
                                 'kernel': node['kernel'], 'cupti_mean_us': node['mean'],
                                 'cupti_p10_us': node['p10'], 'cupti_p90_us': node['p90']})
            modes[mode] = {'label': LABELS[mode], 'host': host,
                          'process_host_geomean_us': [p['timing'][mode]['host_us']['geomean']
                                                     for p in condition['processes']],
                          'cuda_outer': data['baseline_cuda_us'], 'categories': dict(cats),
                          'branches_by_layer': dict(layers),
                          'active_us': trace['mean_active_us'], 'span_us': trace['mean_span_us'],
                          'gap_us': trace['mean_gap_us'],
                          'event_slowdown': data['event_instrumentation_cuda_mean_ratio'],
                          'kernels_per_graph': trace['replays'][0]['kernel_count']}
            timing_rows.append({'condition': key, 'model': size, 'kappa': pair.replace('0p1', '0.1'),
                'gate': '8x16 corner tile' if size == '14M' else 'elementwise one-sided',
                'tiled_threshold': size == '14M', 'mode': mode, 'host_geomean_us': host['geomean'],
                'host_median_us': host['median'], 'host_p10_us': host['p10'], 'host_p90_us': host['p90'],
                'gpu_active_us': trace['mean_active_us'], 'trace_span_us': trace['mean_span_us'],
                'trace_gap_us': trace['mean_gap_us']})
        losses = condition['processes'][0]['loss']
        output['conditions'].append({'key': key, 'model': size, 'kappa': pair.replace('0p1', '0.1'),
            'tiled_threshold': size == '14M', 'source_run': source.name,
            'gpu_uuid': next(iter(uuids)), 'selected_sparse': selected, 'selected_no_skip': no_skip,
            'modes': modes, 'native_loss': losses['native'], 'sparse_loss': losses['baseline_' + selected],
            'logit_comparisons': sum(p['logit_comparisons'] for p in condition['processes']),
            'max_absolute_logit_error': max(p['max_logit_absolute_error'] for p in condition['processes']),
            'max_relative_l2': max(p.get('max_logit_relative_l2', p.get('max_relative_l2')) for p in condition['processes']),
            'consumer_counters': condition.get('consumer_counters')})
    assert sum(c['logit_comparisons'] for c in output['conditions']) == 44616
    assert len({c['gpu_uuid'] for c in output['conditions'] if c['model'] != '14M'}) == 1
    assert len({c['gpu_uuid'] for c in output['conditions']}) == 2
    output['total_logit_comparisons'] = 44616
    output['checkpoint_sources'] = {
        '14M': read(REPO / 'runs/063-2026-10-07-pythia14m-tile-specialized-kernel/provenance/inputs.json')['checkpoints'],
        '31M/70M': read(RUNBIG / 'provenance/inputs.json')['checkpoints']}
    write(ROOT / 'results/decomposition.json', output)
    for filename, rows in [('full-forward-latency.csv', timing_rows), ('forward-stages.csv', stage_rows),
                           ('forward-categories.csv', category_rows), ('branch-bundle-by-layer.csv', layer_rows),
                           ('cuda-kernel-nodes.csv', node_rows)]:
        csvout(filename, rows)
    sources = [source / name for source in [RUN14, RUNBIG]
               for name in ['results/decomposition.json', 'config.json', 'README.md']]
    sources += [REPO / 'runs/060-2026-10-04-pythia14m-temporal-corner-tile/tile_gate.py',
                REPO / 'runs/063-2026-10-07-pythia14m-tile-specialized-kernel/provenance/inputs.json',
                RUNBIG / 'provenance/inputs.json', RUNBIG / 'bindings.py']
    write(ROOT / 'provenance/sources.json', {'sources': [record(p) for p in sources],
        'analysis_source_commit': __import__('subprocess').check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
        'question': 'Package existing per-model forward decomposition with explicit gate and sparse execution identities.'})
    print(f'Collected {len(output["conditions"])} model/threshold points, {len(timing_rows)} execution variants, {len(stage_rows)} forward-stage rows.')


if __name__ == '__main__':
    collect()

"""Add measured Run051/052 executions; component-only candidates have no points."""
import hashlib
import json
import math
from pathlib import Path

from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[2]
R51 = ROOT / 'runs/051-2026-09-22-pythia70m-frozen-kernel-grid'
R52 = ROOT / 'runs/052-2026-09-23-pythia70m-sparse-attribution'
SOURCES = {
    'runs/051-2026-09-22-pythia70m-frozen-kernel-grid/results/complete-table.json':
        '2cbbd5e99ce7d0ea426ff9e1648085027f8a3ef20343645eb29d2653a7c63df0',
    'runs/051-2026-09-22-pythia70m-frozen-kernel-grid/results/final-summary.json':
        '037f1cd55fa78b15518c9c3c55c92ece81439cea29683ad8122d17b626af4e86',
    'runs/052-2026-09-23-pythia70m-sparse-attribution/results/references-001-summary.json':
        '1f5c7a494fb725f6bcf1149cd6984a90d4f98ea4e442abc93fd1d15dbd98d4b1',
}


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def load(existing):
    for path, digest in SOURCES.items():
        assert sha(ROOT / path) == digest, path
    table, summary, repeated = [read(ROOT / path) for path in SOURCES]
    assert table['coverage']['blocks'] == summary['coverage']['blocks'] == 338
    assert table['coverage']['timing_inputs'] == 64 and table['coverage']['timing_passes'] == 7
    evidence = {}
    for root, records in ((R51, summary['sources']), (R52, repeated['sources'])):
        for item in records:
            path = root / item['path']
            assert sha(path) == item['sha256'], path
            evidence[path.relative_to(ROOT).as_posix()] = item['sha256']
    keys = {'Base': ('0', 'none'), 'T2/Ph': ('hz', 'h'), 'T7/Pall': ('7', 'all')}
    by_id = {r['id']: next(old for old in existing if (old['scope'], old['pressure']) == keys[r['recipe']]
                          and old['kappa'] == r['kappa']) for r in table['rows']}
    identities = {}
    for row in table['rows']:
        cid = row['id']; old = by_id[cid]
        checkpoint = read(R51 / f'artifacts/final-001-{cid}-r1/result.json')['checkpoint']
        attempt = ROOT / old['source_attempt']
        manifest_path = attempt / 'manifest.json'
        manifest = read(manifest_path)
        final = attempt / manifest['checkpoints']['final']['path']
        verified = {}
        # Some old aggregate keys include the optimizer and others only model files.
        # Match the actual model/config/metadata bytes, rather than equating hash scopes.
        for item in checkpoint['files']:
            path = final / Path(item['path']).name
            assert path.stat().st_size == item['bytes'] and sha(path) == item['sha256'], path
            verified[path.relative_to(ROOT).as_posix()] = item['sha256']
        evidence[manifest_path.relative_to(ROOT).as_posix()] = sha(manifest_path)
        identities[cid] = dict(plot_checkpoint_key=old['checkpoint_key'],
                               runtime_checkpoint_key=checkpoint['final_checkpoint_content_sha256'],
                               logical_pass_FP16_loss=checkpoint['canonical_logical_products']['coverage']['loss'],
                               verified_model_files=verified)
    rows = []
    for r in table['rows']:
        old = by_id[r['id']]
        assert r['canonical_FP16_loss'] == identities[r['id']]['logical_pass_FP16_loss']
        if 'logical_pass_loss' in old:
            assert math.isclose(r['canonical_FP16_loss'], old['logical_pass_loss'], rel_tol=1e-12)
        modes = ('native_graph', 'native_hz_graph') if r['id'] == 'c00' else ('sparse_c_graph',)
        for mode in modes:
            execution = summary['results'][r['id']]['implementations'][mode]
            assert execution['qualified'] == r['qualified'][mode]
            assert execution['geomean_host_ms'] == r['latency_ms'][mode]
            assert len(execution['process_ms']) == 3
            assert math.isclose(math.exp(math.fsum(map(math.log, execution['process_ms'])) / 3),
                                execution['geomean_host_ms'], rel_tol=1e-12)
            for rep in range(1, 4):
                result = read(R51 / f'artifacts/final-001-{r["id"]}-r{rep}/result.json')
                assert result['status'] == 'complete' and result['validation_blocks'] == 338
                assert result['checkpoint']['final_checkpoint_content_sha256'] == identities[r['id']]['runtime_checkpoint_key']
            rows.append(dict(session='Run051', id=r['id'], model='70M', recipe=r['recipe'],
                             kappa=r['kappa'], backend=mode, loss=old['loss'],
                             checkpoint_key=old['checkpoint_key'], latency_ms=r['latency_ms'][mode],
                             process_latency_ms=execution['process_ms'], qualified=execution['qualified'],
                             logical_pass_FP16_loss=r['canonical_FP16_loss'],
                             native_BF16_loss=r['native_BF16_loss']))
    for cid in ('c00', 'c24', 'c25'):
        old = by_id[cid]
        processes = repeated['records'][cid]
        assert len(processes) == 3
        modes = ('native_graph', 'native_hz_graph') if cid == 'c00' else ('prior_c_graph', 'dense_policy_graph')
        for rep in range(1, 4):
            result = read(R52 / f'artifacts/references-001-{cid}-r{rep}/result.json')
            assert result['status'] == 'complete' and result['validation_blocks'] == 338
            assert result['checkpoint']['final_checkpoint_content_sha256'] == identities[cid]['runtime_checkpoint_key']
        for mode in modes:
            assert all(r['qualified'][mode] for r in processes)
            values = [r['latency_ms'][mode] for r in processes]
            rows.append(dict(session='Run052', id=cid, model='70M',
                             recipe='Base' if cid == 'c00' else 'T2/Ph', kappa=old['kappa'],
                             backend=mode, loss=old['loss'], checkpoint_key=old['checkpoint_key'],
                             latency_ms=math.exp(math.fsum(map(math.log, values)) / 3),
                             process_latency_ms=values, qualified=True,
                             logical_pass_FP16_loss=old['logical_pass_loss'],
                             native_BF16_loss=processes[0]['loss']['native']))
    assert len(rows) == 18 and sum(not r['qualified'] for r in rows) == 2
    return dict(points=rows, sources_sha256=SOURCES, raw_evidence_sha256=evidence,
                checkpoint_identity=identities,
                module_sha256=sha(Path(__file__)),
                loss_convention='Reuse the original ordinary FP16 loss for the identical checkpoint. Logical-pass FP16 and native BF16 losses are retained separately, not substituted on the x-axis.',
                interpretation='C is the retained sparse-h/dense-z policy, not a promoted Run052 experimental candidate. Separate sessions remain separate absolute measurements. Failed numerical measurements are marked and excluded from frontier eligibility.',
                omitted='Run052 E/F/CTA experimental candidates have component timings only; no full-model latency is inferred.')


def draw(ax, data, colors):
    rows = data['points']
    handles = []
    for recipe, label in [('T2/Ph', r'$T_2/P_h$'), ('T7/Pall', r'$T_7/P_{\mathrm{all}}$')]:
        points = sorted((r for r in rows if r['session'] == 'Run051' and r['recipe'] == recipe),
                        key=lambda r: r['kappa'])
        color = colors[recipe]
        # NaNs break the curve at rejected checkpoints; do not bridge the missing .1 point.
        ax.plot([p['loss'] for p in points],
                [p['latency_ms'] if p['qualified'] else float('nan') for p in points],
                color=color, ls=(0, (1.3, 1.8)), lw=1.4, zorder=8)
        for p in points:
            ax.plot(p['loss'], p['latency_ms'], marker='D' if p['qualified'] else 'x',
                    ms=6.2 if p['qualified'] else 6., mfc='white', mec=color,
                    mew=1.3, ls='none', zorder=9)
        handles.append(Line2D([], [], color=color, ls=(0, (1.3, 1.8)), lw=1.4,
                              marker='D', ms=6.2, mfc='white', mec=color, mew=1.3,
                              label=f'Run051 {label} (C)'))
    blue = colors['T2/Ph']
    for mode, marker, label in [('prior_c_graph', '+', r'Run052 $T_2/P_h$ (C)'),
                               ('dense_policy_graph', 'v', r'Run052 $T_2/P_h$ (dense)')]:
        points = sorted((r for r in rows if r['session'] == 'Run052' and r['backend'] == mode),
                        key=lambda r: r['kappa'])
        style = dict(marker=marker, ms=8.5 if marker == '+' else 5.8, mfc='white',
                     mec=blue, mew=1.5 if marker == '+' else 1., ls='none', color=blue)
        ax.plot([p['loss'] for p in points], [p['latency_ms'] for p in points], **style, zorder=10)
        handles.append(Line2D([], [], **style, label=label))
    for session, marker in [('Run051', '^'), ('Run052', 'P')]:
        for mode, label in [('native_graph', 'PyTorch'), ('native_hz_graph', 'dense')]:
            point, = [r for r in rows if r['session'] == session and r['recipe'] == 'Base' and r['backend'] == mode]
            style = dict(marker=marker, ms=7.7, mfc='#52565C' if mode == 'native_graph' else 'white',
                         mec='#52565C', mew=1.2, ls='none', color='#52565C')
            ax.plot(point['loss'], point['latency_ms'], **style, zorder=10)
            handles.append(Line2D([], [], **style, label=f'{session} Base ({label})'))
    handles.append(Line2D([], [], marker='x', color=colors['T7/Pall'], ls='none',
                          ms=6., mew=1.3, label='C: failed validation'))
    # Column-major legend: recipes; dense controls; remaining Base references/failures.
    return [handles[i] for i in (0, 1, 2, 4, 5, 3, 6, 7, 8)]

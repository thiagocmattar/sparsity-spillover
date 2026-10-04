"""Place two corner-tile endpoints on the retained manuscript scale frontier."""
import hashlib
import json
import math
from pathlib import Path
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import NullLocator, ScalarFormatter

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RUN = ROOT / 'runs/061-2026-10-04-pythia14m-corner-tile-latency'
SCALE = ROOT / 'analyses/038-2026-09-24-base-t2-t7-scale-frontier'
PINNED = {
    SCALE / 'data/scale-figure.json': 'c352e3f9073ea2996172b32b6ea31880e8f078c629e7560a3becdc71184efcd8',
    HERE / 'results/summary.json': 'fdf9875e1dfdcc1a30949628fb40905a1987d3766072f55a6b36243733ad66a2',
    RUN / 'results/results.json': '6e9f0aeffdc4d587f766673b1240b2ed08a20ebe2fb9b54a563054d7db3b1b74',
    HERE / 'data/source-identity.json': '6a06041e4c830710eaa13164169b0c0c97938ea73ede542f24529641a91e049b',
}
COLORS = {'hz': '#173FAD', '7': '#C27539'}
TILE = '#008A79'
MARKERS = {'14M': 'o', '31M': '^', '70M': 's'}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def identity(path):
    return dict(path=path.relative_to(ROOT).as_posix(), bytes=path.stat().st_size, sha256=sha(path))


def checked(row):
    path = ROOT / row['path']
    assert identity(path) == row, row['path']


def dominates(a, b):
    return (a['loss'] <= b['loss'] and a['latency_ms'] <= b['latency_ms']
            and (a['loss'] < b['loss'] or a['latency_ms'] < b['latency_ms']))


def frontier(points):
    return sorted([p for p in points if not any(dominates(q, p) for q in points)], key=lambda p: p['loss'])


def collect():
    for path, digest in PINNED.items():
        assert sha(path) == digest, path
    historical = read(SCALE / 'data/scale-figure.json')
    for name, digest in historical['source_sha256'].items():
        assert sha(ROOT / name) == digest, name
    manuscript = ROOT / 'manuscript/draft/figures/02-14m-70m-latency-quality.pdf'
    assert sha(manuscript) == historical['output_sha256']
    assert sha(SCALE / historical['output']) == historical['output_sha256']
    for row in read(HERE / 'data/source-identity.json')['files']:
        checked(row)
    for row in read(RUN / 'transfer/inventory-002.json')['files']:
        path = RUN / row['path']
        assert path.stat().st_size == row['bytes'] and sha(path) == row['sha256']
    quality = read(HERE / 'results/summary.json')
    assert quality['coverage'] == dict(documents=500, blocks=338, input_tokens=692224, predictions=691886, excluded_tail=1444)
    timing = read(RUN / 'results/results.json')
    assert timing['status'] == 'verified'
    historical_points = historical['executions']
    assert len(historical_points) == 36
    new = []
    controls = []
    sources = [identity(path) for path in PINNED]
    sources += [identity(manuscript), identity(SCALE / historical['output'])]
    for pair in timing['pairs']:
        assert pair['qualified'] and pair['paired_samples_per_comparison'] == 1344
        assert len(pair['processes']) == 3
        for source in pair['source_files']:
            path = RUN / source['path']
            assert path.stat().st_size == source['bytes'] and sha(path) == source['sha256']
            sources.append(identity(path))
        for family, kind in [('scalar', 'elementwise'), ('tile', 'corner_tile')]:
            q = next(row for row in quality['rows'] if row['kind'] == kind and row['kappa'] == pair['kappa'])
            process_latencies = []
            for process in pair['processes']:
                attempt = RUN / 'artifacts/attempts' / process['attempt']
                result = read(attempt / 'result.json')
                checkpoint = result['checkpoints'][family]
                assert result['qualified'] and result['status'] == 'complete'
                assert checkpoint['id'] == q['condition'] and checkpoint['source'] == q['attempt']
                assert checkpoint['dose'] == q['kappa']
                samples = [s['host_ms'] for s in read(attempt / 'timing.json')['samples'] if s['mode'] == family + '_sparse']
                assert len(samples) == 448
                process_latencies.append(math.exp(statistics.mean(math.log(s) for s in samples)))
            value = pair['host_geomean_ms'][family + '_sparse']
            assert math.isclose(value, math.exp(statistics.mean(math.log(s) for s in process_latencies)), rel_tol=1e-12)
            point = dict(model='14M', scope='hz', pressure='h', kappa=q['kappa'], loss=q['loss'],
                         latency_ms=value, backend='Run061 sparse', gate=kind, condition=q['condition'],
                         source_attempt=q['attempt'], process_latency_ms=process_latencies,
                         final_checkpoint_content_sha256=checkpoint['final_checkpoint_content_sha256'],
                         quality_precision='canonical FP16', latency_precision='BF16',
                         role='new tile endpoint' if family == 'tile' else 'same-session scalar reference')
            (new if family == 'tile' else controls).append(point)
    assert len(new) == len(controls) == 2
    points = historical_points + new
    measured = frontier(points)
    dominance = []
    for p in new:
        witnesses = [q for q in historical_points if dominates(q, p)]
        dominance.append(dict(condition=p['condition'], nondominated=p in measured, historical_dominators=witnesses))
    return dict(historical_points=historical_points, new_tile_points=new, same_session_scalar_references=controls,
                measured_frontier=measured, historical_frontier=frontier(historical_points),
                tile_dominance=dominance, source_files=sources, historical_coordinates_unchanged=True,
                cohort=dict(historical_execution_points=36, new_tile_points=2, frontier_cohort_points=38,
                            auxiliary_scalar_references=2, training_seed=1234),
                coverage=quality['coverage'],
                quality='canonical full-validation FP16 loss; smaller is better',
                latency='synchronized-host geometric mean; RTX5090 BF16 B1/T2048 uncached full50304 logits',
                caveat='Historical and Run061 sessions/kernel variants differ. Raw coordinates are not normalized. Frontier is descriptive, not a same-kernel causal comparison or significance test. Auxiliary scalar references are excluded from frontier membership.')


def draw_panel(ax, data, scales, detail=False):
    old = [p for p in data['historical_points'] if p['model'] in scales]
    for scale in scales:
        for scope in ['hz', '7']:
            group = sorted([p for p in old if p['scope'] == scope and p['model'] == scale], key=lambda p: p['kappa'])
            ax.plot([p['loss'] for p in group], [p['latency_ms'] for p in group],
                    color=COLORS[scope], ls=(0, (3, 2)) if scope == 'hz' else '-', lw=1.4,
                    marker=MARKERS[scale], ms=5.5, mec='white', mew=.65, zorder=4)
        base = [p for p in old if p['scope'] == '0' and p['model'] == scale]
        for p in base:
            kernel = p['backend'] == 'kernel'
            ax.scatter(p['loss'], p['latency_ms'], marker=MARKERS[scale], s=70 if kernel else 24,
                       facecolor='white' if kernel else '#52565C', edgecolor='#52565C',
                       linewidth=1.1, zorder=6)
        ax.axvline(base[0]['loss'], color='#A1A5AA', lw=.6, ls=(0, (2, 3)), zorder=1)
    points = old + data['new_tile_points']
    edge = frontier(points)
    ax.step([p['loss'] for p in edge], [p['latency_ms'] for p in edge], where='post',
            color='#656C73', ls=(0, (1.5, 2.5)), lw=1.1, zorder=2)
    for p in data['new_tile_points']:
        ax.errorbar(p['loss'], p['latency_ms'],
                    yerr=[[p['latency_ms'] - min(p['process_latency_ms'])],
                          [max(p['process_latency_ms']) - p['latency_ms']]],
                    fmt='D', color=TILE, ms=8, mec='white', mew=.8, capsize=3, elinewidth=1, zorder=9)
        if detail:
            offset = (12, 13) if p['kappa'] == 0 else (12, -16)
            ax.annotate(f"Tile κ={p['kappa']:g}", (p['loss'], p['latency_ms']), xytext=offset,
                        textcoords='offset points', fontsize=9, color=TILE,
                        bbox=dict(facecolor='white', edgecolor='none', pad=.5))
    if detail:
        for p in data['same_session_scalar_references']:
            ax.scatter(p['loss'], p['latency_ms'], marker='D', s=48, facecolor='white',
                       edgecolor=TILE, linewidth=1.3, zorder=8)
        witness = next(p for p in old if p['scope'] == 'hz' and p['kappa'] == .5)
        ax.annotate('Previous scalar κ=0.5', (witness['loss'], witness['latency_ms']), xytext=(-18, -30),
                    textcoords='offset points', ha='center', fontsize=8.5, color=COLORS['hz'],
                    arrowprops=dict(arrowstyle='-', color=COLORS['hz'], lw=.6),
                    bbox=dict(facecolor='white', edgecolor='none', pad=.6))
    ax.set_xlabel('Validation loss (FP16; lower is better)')
    ax.set_ylabel('Full-model latency (ms; lower is better)')
    ax.grid(axis='y', color='#E8EAED', lw=.6)
    ax.set_axisbelow(True)


def main():
    preserved = {p.name: sha(p) for p in (HERE / 'figures').glob('*.pdf')
                 if p.name != '04-quality-latency-frontier-with-corner-tiles.pdf'}
    data = collect()
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9.5, 'pdf.fonttype': 42,
                         'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': .65})
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), gridspec_kw={'width_ratios': [1.1, 1]})
    fig.subplots_adjust(left=.075, right=.985, bottom=.27, top=.87, wspace=.27)
    draw_panel(axes[0], data, ['14M', '31M', '70M'])
    draw_panel(axes[1], data, ['14M'], detail=True)
    axes[0].set(xscale='log', yscale='log', xlim=(4.02, 5.86), ylim=(.43, 2.85))
    axes[0].set_xticks([4.2, 4.5, 4.8, 5.1, 5.4, 5.7])
    axes[0].set_yticks([.5, 1, 1.5, 2, 2.5])
    for axis in [axes[0].xaxis, axes[0].yaxis]:
        axis.set_major_formatter(ScalarFormatter())
        axis.set_minor_locator(NullLocator())
    axes[1].set(xlim=(5.09, 5.85), ylim=(.45, .675))
    axes[1].set_xticks([5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8])
    axes[1].set_yticks([.45, .50, .55, .60, .65])
    axes[0].set_title('(a) Previous manuscript cohort + two tile endpoints', loc='left', fontsize=10)
    axes[1].set_title('(b) 14M detail with matched Run061 scalar references', loc='left', fontsize=10)
    fig.suptitle('Quality-latency frontier with 14M corner tiles', y=.98, fontsize=14)
    handles = [
        Line2D([], [], marker='o', ls='none', mfc='white', mec='#52565C', ms=8, label='Base (kernel)'),
        Line2D([], [], marker='o', ls='none', color='#52565C', ms=5, label='Base (PyTorch)'),
        Line2D([], [], color=COLORS['hz'], ls=(0, (3, 2)), label=r'Previous $T_2/P_h$'),
        Line2D([], [], color=COLORS['7'], label=r'Previous $T_7/P_h$'),
        Line2D([], [], marker='D', ls='none', color=TILE, ms=7, label=r'Tile $T_2/P_h$ (Run061)'),
        Line2D([], [], marker='D', ls='none', mfc='white', mec=TILE, ms=7, label='Matched scalar (Run061; reference)'),
        Line2D([], [], color='#656C73', ls=(0, (1.5, 2.5)), label='Observed Pareto frontier'),
    ]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.53, .095), ncol=4,
               frameon=False, fontsize=8.5, columnspacing=1.4, handlelength=2.1)
    model_handles = [Line2D([], [], marker=m, ls='none', color='#52565C', label=s) for s, m in MARKERS.items()]
    fig.legend(handles=model_handles, loc='lower center', bbox_to_anchor=(.53, .049), ncol=3,
               frameon=False, fontsize=8.5, columnspacing=2.3)
    fig.text(.53, .018, 'Frontier: 36 previous points + 2 tiles; hollow scalar references excluded.\n'
             'Whiskers span 3 process means. Sessions/kernels differ; raw coordinates, no normalization.',
             ha='center', fontsize=8, color='#555B61')
    output = HERE / 'figures/04-quality-latency-frontier-with-corner-tiles.pdf'
    fig.savefig(output, bbox_inches='tight', metadata={'Title': 'Quality-latency frontier with 14M corner tiles',
                                                     'CreationDate': None, 'ModDate': None})
    plt.close(fig)
    for name, digest in preserved.items():
        assert sha(HERE / 'figures' / name) == digest
    data.update(output=identity(output), source_script=identity(Path(__file__)),
                preserved_figures=preserved, figure=dict(overview_scale='log/log', detail_scale='linear/linear',
                whiskers='minimum/maximum geometric mean over three fresh processes; not confidence intervals'))
    (HERE / 'data/quality-latency-frontier.json').write_text(json.dumps(data, indent=2) + '\n', encoding='utf-8', newline='\n')
    lines = ['# Corner-tile endpoints in the previous quality-latency frontier', '',
             '| Gate | κ | Canonical FP16 loss | Run061 sparse ms | Process mean range (ms) | On observed frontier? |',
             '|---|---:|---:|---:|---|---|']
    for p in data['new_tile_points'] + data['same_session_scalar_references']:
        role = ('Yes' if p in data['measured_frontier'] else 'No') if p['gate'] == 'corner_tile' else 'Reference only'
        lines.append(f"| {p['gate']} | {p['kappa']:g} | {p['loss']:.6f} | {p['latency_ms']:.6f} | "
                     f"{min(p['process_latency_ms']):.6f}–{max(p['process_latency_ms']):.6f} | {role} |")
    lines += ['', 'Historical T2/Ph κ=.5: loss5.536255, latency.506323ms. It dominates both tile endpoints by the retained coordinates.',
              data['caveat']]
    (HERE / 'results/quality-latency-frontier.md').write_text('\n'.join(lines) + '\n', encoding='utf-8', newline='\n')
    print(json.dumps({'historical_points': 36, 'new_tile_points': 2, 'frontier_points': len(data['measured_frontier']),
                      'tile_frontier_members': sum(p in data['measured_frontier'] for p in data['new_tile_points']),
                      'historical_figures_preserved': list(preserved), 'output': str(output)}))


if __name__ == '__main__':
    main()

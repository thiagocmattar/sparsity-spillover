"""Write common semantic budgets and separately identified measured detail."""
import importlib.util
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import labels
import detail_figures
import comparison_figure

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('collect042', ROOT / '01_collect.py')
collect = importlib.util.module_from_spec(spec)
spec.loader.exec_module(collect)
PLOT_LABELS = {'native': 'Native (dense)', 'generic_sparse': 'Generic sparse',
               'generic_no_skip': 'Generic, no skip', 'tile_sparse': 'Corner sparse',
               'tile_no_skip': 'Corner, no skip', 'sparse': 'Scalar sparse',
               'no_skip': 'Scalar, no skip'}


def md(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((text.rstrip() + '\n').encode('utf-8'))


def table(headers, rows):
    return '| ' + ' | '.join(headers) + ' |\n| ' + ' | '.join(['---'] * len(headers)) + ' |\n' + ''.join(
        '| ' + ' | '.join(str(x) for x in row) + ' |\n' for row in rows)


def reduction(reference, value):
    return 100 * (1 - value / reference)


def mode_order(c):
    return ['native', 'generic_sparse', 'generic_no_skip', 'tile_sparse', 'tile_no_skip'] if c['model'] == '14M' else ['native', 'sparse', 'no_skip']


def branch_total(m):
    return sum(value for label, value in m['categories'].items() if label in collect.BRANCH)


def main():
    data = collect.read(ROOT / 'results/decomposition.json')
    conditions = data['conditions']
    models = ['14M', '31M', '70M']
    figures = ROOT / 'figures'
    observations = ROOT / 'observations'
    figures.mkdir(exist_ok=True)
    observations.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size': 9, 'pdf.fonttype': 42})
    maximum_host = max(m['host']['p90'] for c in conditions for m in c['modes'].values()) / 1000 * 1.06
    maximum_active = max(m['active_us'] for c in conditions for m in c['modes'].values()) * 1.04
    maximum_layer = max(v for c in conditions for m in c['modes'].values() for v in m['branches_by_layer'].values()) * 1.08
    coverage = ('BF16, batch 1, 2,048 tokens, resident inputs, uncached full 50,304-way logits. '
                'Each mode/threshold has 64 timing inputs x 7 paired passes x 3 fresh processes '
                '= 1,344 uninstrumented samples. Numerical qualification covers all 500 validation documents, '
                '338 complete blocks (692,224 input tokens; 1,444 excluded tail tokens). '
                'CUPTI uses 4 inputs x 3 replays in process 1, giving 12 traced forwards per mode/threshold. '
                '14M comes from Run064; 31M/70M from Run065. Both use RTX 5090, but the 14M physical GPU/session differs.')
    figure_entries = []
    fig, axes = plt.subplots(3, 2, figsize=(12.5, 8.6), layout='constrained')
    for row, size in enumerate(models):
        for column, c in enumerate(x for x in conditions if x['model'] == size):
            ax = axes[row, column]
            names = mode_order(c)
            for i, name in enumerate(names):
                m = c['modes'][name]
                chosen = name == c['selected_sparse']
                ax.barh(i, m['host']['geomean'] / 1000, color=labels.MODE_STYLES[name]['color'],
                        alpha=1 if chosen else .55, hatch='///' if 'no_skip' in name else '',
                        edgecolor='white', linewidth=.4)
                ax.plot([m['host']['p10'] / 1000, m['host']['p90'] / 1000], [i, i], color='black', linewidth=1.1)
                ax.scatter([v / 1000 for v in m['process_host_geomean_us']], [i-.065, i, i+.065],
                           marker='D', s=9, color='black', zorder=3)
                ax.text(m['host']['p90'] / 1000 + .025, i, f'{m["host"]["geomean"]/1000:.3f}', va='center', fontsize=8)
            ax.set_yticks(range(len(names)), [PLOT_LABELS[n] for n in names])
            ax.invert_yaxis()
            ax.set_xlim(0, maximum_host)
            ax.set_xlabel('Complete forward, synchronized host latency (ms)')
            ax.set_title(f'{size}, kappa = {c["kappa"]}: ' + ('TILED threshold' if c['tiled_threshold'] else 'SCALAR threshold'))
            ax.grid(axis='x', alpha=.2)
    fig.suptitle('Per-model forward latency — threshold rule and sparse execution are separate')
    caption = ('Bars show pooled host geometric means; black whiskers are pooled sample p10-p90; '
               'three black diamonds show fresh-process geometric means. These are descriptive distributions, '
               'not confidence intervals. Blue means native; green the selected sparse implementation; '
               'red the selected no-skip control; gray the additional generic 14M consumer. '
               'Hatching means arithmetic skipping is disabled; the trained gate stays active. All panels share '
               'the millisecond axis. 14M has a tiled gate in every mode, including native and generic. '
               '31M/70M have scalar gates in every mode. Different gates/consumers and physical GPU sessions '
               'make the cross-size comparison descriptive.')
    fig.text(.5, -.016, 'B1 / T2048 / BF16 / RTX 5090; 14M separate session. Whiskers: p10-p90; diamonds: 3 process means.',
             ha='center', fontsize=8)
    name = '01-full-forward-latency'
    fig.savefig(figures / (name + '.pdf'), bbox_inches='tight', pad_inches=.18)
    plt.close(fig)
    figure_entries.append((name, 'How does complete-forward latency vary by model, gate and execution mode?', caption,
                           'The selected sparse implementations reduce latency versus their own gated native references; '
                           'the 14M specialized-versus-generic comparison is distinct from either comparison with native.'))
    for number, size in enumerate(models, 2):
        cs = [c for c in conditions if c['model'] == size]
        fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.6), layout='constrained')
        for column, c in enumerate(cs):
            names = mode_order(c)
            ax = axes[0, column]
            left = [0.] * len(names)
            for category, style in labels.COMMON_STYLES.items():
                vals = [c['modes'][n]['semantic_categories'][category] for n in names]
                ax.barh(range(len(names)), vals, left=left, color=style['color'],
                        hatch=style['hatch'], edgecolor='white', linewidth=.3)
                left = [a+b for a, b in zip(left, vals)]
            ax.set_yticks(range(len(names)), [PLOT_LABELS[n] for n in names])
            ax.invert_yaxis()
            ax.set_xlim(0, maximum_active)
            ax.set_xlabel('Summed actual CUDA activities per forward (us)')
            ax.set_title(f'kappa = {c["kappa"]}: complete GPU-active budget')
            ax.grid(axis='x', alpha=.2)
            ax = axes[1, column]
            comparison = ['native', c['selected_sparse'], c['selected_no_skip']]
            if size == '14M':
                comparison.insert(1, 'generic_sparse')
            for name in comparison:
                values = c['modes'][name]['branches_by_layer']
                ax.plot(range(6), [values[str(j)] for j in range(6)],
                        **labels.MODE_STYLES[name], label=PLOT_LABELS[name])
            ax.set_ylim(0, maximum_layer)
            ax.set_xticks(range(6))
            ax.set_xlabel('Transformer layer')
            ax.set_ylabel('H/Z output block per layer (us)')
            ax.set_title('H/Z output block: gates + preparation + W2 + Wo + residual')
            ax.grid(alpha=.2)
            ax.legend(fontsize=8, loc='upper left')
        handles = [Patch(facecolor=style['color'], hatch=style['hatch'], label=category)
                   for category, style in labels.COMMON_STYLES.items()]
        fig.legend(handles=handles, loc='outside lower center', ncol=3, fontsize=8)
        fig.suptitle(f'{size} — ' + ('TILED 8 x 16 corner threshold' if size == '14M' else 'SCALAR elementwise threshold (not tiled)'))
        name = f'{number:02d}-{size.lower()}-decomposition'
        fig.savefig(figures / (name + '.pdf'), bbox_inches='tight', pad_inches=.18)
        plt.close(fig)
        caption = (f'{size} at kappa 0 and 0.1. Upper panels sum per-stage CUPTI means from 12 traced forwards '
                   'into nine fixed semantic categories. All model sizes and modes use the same category order, '
                   'colors, patterns and complete legend. The purple H/Z output block includes gates, sparse preparation, '
                   'W2, Wo and residual combination, whether fused or separately measured. No fused consumer is '
                   'split into unmeasured internal phases. Bars are activity budgets, not chronological timelines. '
                   'Lower panels show the same H/Z output block per layer; line colors encode execution mode: '
                   'blue native, green selected sparse, red selected no-skip, gray additional 14M generic. '
                   'All three model figures share the upper microsecond axis and lower vertical axis. '
                   'The category is an accounting group, not an additional timed kernel. Gaps and host overhead are excluded. '
                   'No event or CUPTI number is rescaled to the primary uninstrumented host total. '
                   'CUPTI means are descriptive; distribution data are retained per original stage/kernel in the CSVs.')
        result = {'14M': 'Corner specialization reduces the fused H/Z output block cost; vocabulary projection and attention lead afterward.',
                  '31M': 'The sparse H/Z output block is slower than the equivalent native block; savings in other stages compensate.',
                  '70M': 'Sparse H down projections become much cheaper than the no-skip control; vocabulary projection leads the remaining active budget.'}[size]
        figure_entries.append((name, f'Where does {size} forward GPU activity go, and what do the H/Z branches cost per layer?', caption, result))
    figure_entries.extend(detail_figures.build(data, ROOT, mode_order, PLOT_LABELS))
    figure_entries.append(comparison_figure.build(data, ROOT))
    md(ROOT / 'LABELS.md', labels.dictionary_text())
    summary = []
    for c in conditions:
        m = c['modes']; s = m[c['selected_sparse']]; n = m['native']; off = m[c['selected_no_skip']]
        summary.append([c['model'], 'TILED 8x16' if c['tiled_threshold'] else 'SCALAR (not tiled)', c['kappa'],
                        f'{n["host"]["geomean"]/1000:.4f}', f'{s["host"]["geomean"]/1000:.4f}',
                        f'{off["host"]["geomean"]/1000:.4f}',
                        f'{reduction(n["host"]["geomean"], s["host"]["geomean"]):.2f}%',
                        f'{reduction(off["host"]["geomean"], s["host"]["geomean"]):.2f}%'])
    text = [
        '# Latency decomposition by model: tiled versus scalar thresholds', '',
        '**Only the 14M checkpoints in this analysis have a tiled threshold. The 31M and 70M checkpoints use elementwise scalar thresholds.** Kernel tiles, token groups, feature unions and sparse matrix arithmetic do not imply a tiled threshold.', '',
        'This analysis packages the completed Run064 and Run065 measurements requested by the user. It performs local reduction and plotting of saved evidence. The source runs, checkpoints, gate rules and execution policies remain unchanged. There is no new GPU benchmark, training or manuscript edit.', '',
        '## Gate identity and execution identity', '',
        'T2 means the gate topology **h and z**; it is not a tile-size label. Ph means the retained training intervention used OL1 pressure at h. At h, the gate replaces GELU; z is the concatenated attention context immediately before the attention output projection. All six checkpoints were retained after 712 updates; the present work measures inference.', '',
        table(['Model', 'Checkpoint source', 'Gate rule at h and z', 'Tiled threshold?', 'Selected sparse consumer'], [
            ['14M', 'Run060', 'Keep an entire 8-token x 16-feature tile iff its top-left value > kappa; otherwise zero the whole tile', '**YES**', 'Run063 corner-specialized joint h/z consumer'],
            ['31M', 'Run054', 'Keep each element x iff x >= kappa; otherwise zero that element', '**NO — scalar**', 'Frozen opt073-31m-v1 joint h/z consumer'],
            ['70M', 'Run043', 'Keep each element x iff x >= kappa; otherwise zero that element', '**NO — scalar**', 'Frozen Run051 sparse_c; gathered h in layers 1–5']]), '',
        'The 14M tile rule rejects equality at the corner and preserves every signed interior value in a surviving tile, including negative/subthreshold values. The scalar rule keeps equality and applies independently to every element. Comparisons use the activation comparison dtype. At kappa = 0 the scalar rule is elementwise ReLU, while the tile rule still retains signed interiors when the corner survives.', '',
        '**Native is dense arithmetic on the same gated checkpoint, not an ungated A0/dense-trained model.** A no-skip control still applies the trained threshold and produces its zeros; it disables arithmetic skipping in the chosen consumer. At 70M the control retains compaction and small-group padding, so it is not a best-dense-kernel reference.', '',
        table(['Model / execution variant', 'Tiled threshold?', 'What consumes the gated activations?'], [
            ['14M native', 'YES', 'Dense projections with the same corner gate'],
            ['14M generic_sparse', 'YES', 'Generic joint consumer inspects activation tiles and skips eligible work'],
            ['14M tile_sparse', 'YES', 'Specialized joint consumer reads tile corners directly and skips eligible h/z work'],
            ['14M generic_no_skip / tile_no_skip', 'YES', 'Same corresponding consumer with h/z skipping disabled'],
            ['31M native', 'NO', 'Dense projections with scalar h/z gates'],
            ['31M sparse', 'NO', 'Short-row prepass plus joint h/z fallback; eligible rows/tiles skip work'],
            ['31M no_skip', 'NO', 'Joint fallback with skipping disabled; prepass omitted'],
            ['70M native', 'NO', 'Dense projections with scalar h/z gates'],
            ['70M sparse', 'NO', 'h0 is dense; h1–5 compact feature unions and use gathered products; all z products stay dense'],
            ['70M no_skip', 'NO', 'Same scalar gates and conversion; h consumers compute the full K range']]), '',
        '## Complete-forward latency', '',
        coverage, '',
        'Primary numbers below are pooled geometric means of synchronized host latency in milliseconds. The selected sparse mode is the 14M corner-specialized consumer and the existing scalar-threshold consumer at 31M/70M. Reductions use each point’s own reference.', '',
        table(['Model', 'Threshold rule', 'kappa', 'Native ms', 'Selected sparse ms', 'Selected no-skip ms', 'Reduction vs native', 'Reduction vs no-skip'], summary), '',
        '[Figure 1 — full-forward latency, distributions and process repeats](figures/01-full-forward-latency.pdf)', '',
        figure_entries[0][2], '',
        '**The 14M specialization gain is a different comparison:** against the generic consumer on the same tiled checkpoint, the corner consumer reduces host latency by ' +
        '/'.join(f'{reduction(c["modes"]["generic_sparse"]["host"]["geomean"], c["modes"]["tile_sparse"]["host"]["geomean"]):.2f}%' for c in conditions if c['model'] == '14M') +
        ' at kappa 0/0.1. The larger native-to-sparse reductions also include frozen norm, layout, attention, head and dispatch optimizations. They cannot all be attributed to sparsity.', '',
        '## Complete forward map', '',
        'Position indices → token embedding → RoPE tables → six transformer blocks → final normalization → 50,304-way vocabulary projection. Each block has parallel attention/MLP residual topology, but its measured GPU stages execute serially on one stream.', '',
        table(['Step', 'Native dense execution', '14M corner / 31M scalar optimized execution', '70M scalar optimized execution'], [
            ['1', 'Separate attention and MLP layer norms', 'Paired norms', 'Paired norms'],
            ['2', 'QKV projection', 'QKV projection', 'QKV projection'],
            ['3', 'RoPE and QKV layout', 'Fused RoPE/layout', 'Fused RoPE/layout'],
            ['4', 'Causal attention (QK, softmax, PV)', 'Fused attention; split/combine activities mapped', 'Fused attention; split/combine activities mapped'],
            ['5', 'Context layout, z gate, z down projection', 'Context layout; raw z reaches the joint consumer', 'Context layout; z scalar gate fused with dense z down projection'],
            ['6', 'MLP up projection, h gate, h down projection', 'MLP up projection; raw h reaches the joint consumer', 'MLP up projection; h0 gate + dense down, h1–5 compaction + gathered down'],
            ['7', 'Attention + MLP + residual sum', 'Joint gate/down/residual consumer (31M includes prepass + fallback)', 'BF16 residual combination']]), '',
        'The joint stage and fused attention remain indivisible where the evidence has no independent wall-time boundary. Their ordered CUDA activities are available in the kernel inventory; internal SM-cycle probes from Run064 are not additive device latencies. [Every original forward stage](results/forward-stages.csv) and [every ordered CUDA activity](results/cuda-kernel-nodes.csv) retain finer detail.', '',
        '## Per-model decomposition', '',
        'Tables and Figures 2–4 and 8 use nine common semantic categories with identical labels, colors, patterns and ordering for every model and execution mode. The purple **H/Z output block** is gates + sparse preparation + W2 + Wo + residual sum. At 14M/31M it includes the whole indivisible fused consumer; at 70M/native it sums the matching separately measured stages. This grouping preserves the original activity total and does not manufacture internal h/z timings. [Exact definitions, colors and source-stage mapping (Portuguese)](LABELS.md).', '',
        'Tables report arithmetic means of actual CUPTI CUDA activity durations, summed over all six layers where applicable. Shares divide by that mode’s GPU-active interval union; they are not percentages of uninstrumented host latency. The bars are accounting budgets, not timelines. Original implementation-stage values remain in the detailed CSVs and Figures 5–7. In those details, N/A means no independent measurement; an observed zero context-layout activity means a view.', '']
    for number, size in enumerate(models, 2):
        text += [f'### {size}: ' + ('TILED threshold, corner-specialized sparse consumer' if size == '14M' else 'SCALAR threshold, not tiled'), '',
                 f'[Figure {number} — {size} complete GPU budget and per-layer branch costs](figures/{number:02d}-{size.lower()}-decomposition.pdf)', '',
                 figure_entries[number-1][2], '']
        for c in (x for x in conditions if x['model'] == size):
            selected = c['selected_sparse']; names = ['native', selected, c['selected_no_skip']]
            ms = c['modes']; active = ms[selected]['active_us']
            rows = []
            for category in labels.COMMON_STYLES:
                rows.append([category] + [f'{ms[n]["semantic_categories"][category]:.3f}' for n in names] +
                            [f'{ms[selected]["semantic_categories"][category]/active*100:.2f}%'])
            text += [f'#### kappa = {c["kappa"]}', '',
                     table(['Common semantic category', 'Native us', 'Selected sparse us', 'No-skip us', 'Selected sparse active share'], rows), '',
                     'The same H/Z output block (gates + sparse preparation + W2 + Wo + residual), by layer:', '',
                     table(['Layer'] + [PLOT_LABELS[n] + ' us' for n in names],
                           [[j] + [f'{ms[n]["branches_by_layer"][str(j)]:.3f}' for n in names] for j in range(6)]), '']
        cs = [c for c in conditions if c['model'] == size]
        if size == '14M':
            text += ['The specialized H/Z output block is cheaper than the generic H/Z output block on the same tiled activations. Both are fused implementations of the same semantic category. The remaining selected-sparse GPU-active budget is led by the vocabulary projection (38.31%/40.20%) and attention (25.75%/27.12%). The checkpoints’ previously measured quality penalty from training with tile gates is unchanged; faster inference does not repair it.', '',
                     table(['kappa', 'Generic sparse host ms', 'Corner sparse host ms', 'Generic H/Z output us', 'Corner H/Z output us'],
                           [[c['kappa'], f'{c["modes"]["generic_sparse"]["host"]["geomean"]/1000:.4f}',
                             f'{c["modes"]["tile_sparse"]["host"]["geomean"]/1000:.4f}',
                             f'{branch_total(c["modes"]["generic_sparse"]):.3f}', f'{branch_total(c["modes"]["tile_sparse"]):.3f}'] for c in cs]), '']
        elif size == '31M':
            text += ['The H/Z output block is fused at 31M. It takes 44.43%/36.89% of sparse GPU-active time, and costs 431.56/315.84 us versus 172.88/172.85 us for the complete native H/Z output block. Savings in other optimized stages and dispatch gaps compensate for this slower consumer. The kappa 0 short-row prepass completes no token row over the entire 338-block validation workload; kappa 0.1 helps mainly layers 1–3.', '']
        else:
            text += ['Vocabulary projection is the largest remaining sparse component (36.55%/38.01%); attention takes 18.17%/18.92%. Actual h feature-union counters in the five gathered layers retain 7.815%/3.052% of the no-skip padded tensor-core logical products. Those counters are not measured speedups. All z products and h0 stay dense.', '']
    text += ['## Measured detail of the H/Z output block', '',
             'These are detail views of the same purple category used above. Patterns distinguish measured children within the purple family. A solid fused H/Z output block is indivisible in the available evidence; it is not an H-only projection. At 70M, H down (W2) is one measured child, alongside H preparation, Z gate/down and residual. Their sum is the comparable whole. [Detail data](results/hz-detail.csv) retain only components measured independently.', '']
    for number, entry in enumerate(figure_entries[4:7], 5):
        name, _, caption, _ = entry
        text += [f'[Figure {number} — {name[3:6].upper()} H/Z implementation detail](figures/{name}.pdf)', '', caption, '']
    name, _, caption, _ = figure_entries[7]
    text += ['## Final dense–sparse comparison across model sizes', '',
             f'[Figure 8 — dense versus sparse execution, identical semantic colours](figures/{name}.pdf)', '', caption, '']
    budget_rows, distribution_rows, quality_rows = [], [], []
    for c in conditions:
        for name in mode_order(c):
            m = c['modes'][name]
            distribution_rows.append([c['model'], c['kappa'], name, f'{m["host"]["geomean"]:.3f}',
                                      f'{m["host"]["p10"]:.3f}', f'{m["host"]["p90"]:.3f}',
                                      '/'.join(f'{v:.3f}' for v in m['process_host_geomean_us'])])
        m = c['modes'][c['selected_sparse']]
        budget_rows.append([c['model'], c['kappa'], f'{m["active_us"]:.3f}', f'{m["gap_us"]:.3f}',
                            f'{m["span_us"]:.3f}', f'{m["host"]["geomean"]:.3f}', f'{m["event_slowdown"]:.3f}x'])
        quality_rows.append([c['model'], c['kappa'], f'{c["native_loss"]:.8f}', f'{c["sparse_loss"]:.8f}',
                             f'{c["sparse_loss"]-c["native_loss"]:.3g}', c['logit_comparisons']])
    text += ['## Timing boundaries and qualification', '',
             table(['Model', 'kappa', 'Sparse active us', 'Trace gaps us', 'Trace span us', 'Unprofiled host geomean us', 'Event slowdown'], budget_rows), '',
             'Active time plus gaps closes to the **profiled** first-to-last activity span. The span and unprofiled host timing use different measurement boundaries and profiling overhead, so their difference is not assigned to an invented stage. CUDA-event partition intervals are additive within their own instrumented graphs; events slow execution and are not rescaled into the baseline.', '',
             table(['Model', 'kappa', 'Native validation loss', 'Selected sparse loss', 'Sparse minus native', 'Complete-output comparisons'], quality_rows), '',
             'All 44,616 complete-output comparisons passed the original criteria across 18 fresh profiling processes. Run064 is exact in logits and loss. Run065 passes per-logit atol 0.25 + rtol 0.02, relative L2 0.02 and full-loss atol 0.001; its largest absolute difference is 0.75 and relative L2 is about 0.0010043. A fixed absolute-only 0.25 claim would be incorrect. All 88 input/mode trace mappings succeeded in their source reductions.', '',
             'The rows are different trained checkpoints, gates and consumers, with one training seed. They describe the observed workload; they do not isolate architecture-only scaling, establish a scaling law, test the corner-specialized kernel at 31M/70M, or cover autoregressive decoding, other batching/sequence lengths or hardware. Native-to-sparse includes all frozen component changes. Sparse-to-no-skip measures the net skipping effect within each implementation, including its overhead and padding.', '',
             '### All execution modes and timing spread', '',
             'Microseconds; p10/p90 describe the pooled 1,344 samples and are not confidence bounds.', '',
             table(['Model', 'kappa', 'Mode', 'Host geomean us', 'p10 us', 'p90 us', 'Process 1/2/3 geomean us'], distribution_rows), '',
             '## Evidence and regeneration', '',
             '[Run064 observation](../../runs/064-2026-10-07-pythia14m-forward-latency-decomposition/observations/01-forward-decomposition.md) · [Run065 observation](../../runs/065-2026-10-07-larger-model-forward-decomposition/observations/01-larger-forward-decomposition.md). The checkpoint sources are Run060 (14M), Run054 (31M) and Run043 (70M). Exact source/checkpoint identities and SHA256 values are in [source provenance](provenance/sources.json) and [combined evidence](results/decomposition.json). Figures have individual [observations](observations/INDEX.md).', '',
             '[Full-forward timings](results/full-forward-latency.csv) · [Common semantic categories](results/semantic-categories.csv) · [Category mapping](results/category-mapping.csv) · [Original implementation categories](results/forward-categories.csv) · [H/Z detail](results/hz-detail.csv) · [Every forward stage](results/forward-stages.csv) · [Per-layer H/Z block](results/branch-bundle-by-layer.csv) · [Every CUDA activity](results/cuda-kernel-nodes.csv).', '',
             'Run locally from the repository root:', '',
             '```powershell', f'python analyses/{ROOT.name}/01_collect.py', f'python analyses/{ROOT.name}/02_report.py',
             f'python analyses/{ROOT.name}/03_verify.py', '```', '',
             'These scripts read existing completed evidence and generate this analysis only. Publication figures are PDF only; links above open the figures. No source run is overwritten, no cloud resource is created, and no manuscript/finding promotion is implied.']
    md(ROOT / 'README.md', '\n'.join(text))
    for name, question, caption, result in figure_entries:
        md(observations / (name + '.md'), '\n'.join([
            '# ' + name, '', '**Question:** ' + question, '', '**Method and coverage:** ' + coverage, '',
            '**Sources:** Run064 results/decomposition.json and Run065 results/decomposition.json; exact identities in ../provenance/sources.json. Main tables in ../README.md.', '',
            '**Aggregation:** Host latency uses pooled geometric means and sample quantiles. GPU categories use arithmetic sums of measured CUPTI activity means. The H/Z output block pools disjoint named activities within each layer; the definition is fixed across models. Shares divide by that mode’s measured GPU-active union; no host-time rescaling. Exact definitions and fixed colors are in ../LABELS.md.', '',
            '**Caption and legend:** ' + caption, '', '**Result:** ' + result, '',
            '**Caveats:** One training seed; different checkpoint/gate/kernel across model sizes, and separate physical 14M GPU/session. Tiled threshold is present only at 14M. Native is dense execution on the same gated checkpoint; no-skip retains the gate. No confidence interval, architecture-only scaling claim or independent internal fused-phase wall times are implied.', '',
            '**Source script:** ../02_report.py (data: ../01_collect.py).', '',
            f'[Figure PDF](../figures/{name}.pdf) · [Full tables](../README.md)']))
    md(observations / 'INDEX.md', '# Figure observations\n\n' + '\n'.join(
        f'- [{name}]({name}.md): {question}' for name, question, _, _ in figure_entries))
    paths = [ROOT / 'README.md', ROOT / 'LABELS.md'] + list(observations.glob('*.md')) + list(figures.glob('*.pdf')) + list((ROOT / 'results').glob('*.csv'))
    collect.write(ROOT / 'provenance/generated.json', {
        'scripts': [collect.record(ROOT / '01_collect.py'), collect.record(ROOT / '02_report.py'),
                    collect.record(ROOT / '03_verify.py'), collect.record(ROOT / 'labels.py'),
                    collect.record(ROOT / 'detail_figures.py'), collect.record(ROOT / 'comparison_figure.py')],
        'input': collect.record(ROOT / 'results/decomposition.json'),
        'outputs': [collect.record(p) for p in sorted(paths)]})
    print('Created README.md, LABELS.md, 8 PDF figures/observations and standard semantic timing tables.')


if __name__ == '__main__':
    main()

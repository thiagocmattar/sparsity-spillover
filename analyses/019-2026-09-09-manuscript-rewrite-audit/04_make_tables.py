"""Format audited existing results; requires no checkpoints or inference."""
from pathlib import Path
import hashlib
import json

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PRIOR = ROOT / 'analyses/018-2026-09-08-results-materials'
OUT = HERE / 'tables'
SCALES = ('14M', '70M', '410M')


def tabular(columns, header, rows):
    return '\n'.join([rf'\begin{{tabular}}{{@{{}}{columns}@{{}}}}',
                      r'\toprule', header + r' \\', r'\midrule',
                      *[r + r' \\' for r in rows], r'\bottomrule', r'\end{tabular}'])


def write(name, body):
    (OUT / (name + '.tex')).write_text(
        '% Analysis 019 / 04_make_tables.py; audited existing records only.\n' + body.rstrip() + '\n',
        encoding='utf-8', newline='\n')


def main():
    OUT.mkdir(exist_ok=True)
    training = json.loads((HERE / 'training-audit.json').read_text())
    runtime = json.loads((HERE / 'runtime-audit.json').read_text())
    original = json.loads((PRIOR / 'figure_data.json').read_text())
    assert len(training['endpoints']) == 54 and len(training['paired_contrasts']) == 29
    # Retain the six original numerical fragments without changing any row.
    for name in ('scale-contrasts-compact', 'scale-contrasts-full', 'paired-effects',
                 'density-mass', 'operation-contributions', 'kernel-ablations'):
        body = (PRIOR / 'manuscript-tables' / (name + '.tex')).read_text(encoding='utf-8')
        write(name, body)
    families = ('A0', 'A1-H', 'A1-H-L1', 'A1-H-OL1', 'A4', 'A4-OL1', 'A7', 'A7-OL1')
    rows = []
    for r in sorted(training['endpoints'], key=lambda r: (SCALES.index(r['scale']), families.index(r['family']), r['dose'] or 0)):
        symbol = r'\lambda' if r['family'].startswith('A1-H-') else r'\kappa'
        parameter = '---' if r['dose'] is None else f"${symbol}={r['dose']:g}$"
        rows.append(f"{r['scale']} & {r['family']} & {parameter} & {r['loss']:.4f} & {r['delta_loss_vs_A0']:+.4f} & {100*r['R_model']:.3f} & {100*r['S_block']:.3f}")
    write('trained-endpoints', '\n'.join([
        r'\begin{table}[p]\centering\small\renewcommand{\arraystretch}{0.95}',
        r'\caption{All 54 trained endpoints. $\Delta$ loss uses same-size A0. $\lambda$ is pressure weight; $\kappa$ is the trained threshold. Block-only sparsity $\Sblock=\Smodel/\Rarch(\mathrm{A7})$ uses the same A7 reference within each size, including A0. Rounded zero sparsity need not be exactly zero.}\label{tab:trained-endpoints}',
        tabular('lllrrrr', r'Size & Recipe & Parameter & Loss & $\Delta$ loss & $\Smodel$ (\%) & $\Sblock$ (\%)', rows), r'\end{table}']))
    rows = []
    # Group only budgets whose selected endpoints are exactly identical.
    for scale, budgets in [('14M', (.05, .1, .2)), ('70M', (.05, .1)), ('70M', (.2,)), ('410M', (.05, .1)), ('410M', (.2,))]:
        selected = [next(r for r in training['quality_budgets'] if r['scale'] == scale and r['budget'] == b) for b in budgets]
        r = selected[0]
        assert all((x['best_trained_sparse'] or {}).get('id') == (r['best_trained_sparse'] or {}).get('id') and x['best_all_evaluated']['id'] == r['best_all_evaluated']['id'] for x in selected)
        a, b = r['best_trained_sparse'], r['best_all_evaluated']
        aname = 'none' if a is None else a['family'] + (r' $\lambda=1$' if a['family'] == 'A1-H-L1' else '')
        aval = '---' if a is None else f"{a['sparsity_percent']:.3f}"
        bname = b['family'] + (r' $\lambda=1$' if b['family'] == 'A1-H-L1' else '') + f", $p={b['dose']:g}$"
        rows.append(f"{scale} & {', '.join(f'{x:.2f}' for x in budgets)} & {aname} & {aval} & {bname} & {b['sparsity_percent']:.3f}")
    write('quality-budgets', tabular('lllr lr'.replace(' ', ''),
          r'Size & Loss budget & Trained recipe & $\Smodel$ (\%) & Including clipping & $\Smodel$ (\%)', rows))
    rows = [f"{r['family']} & {r['projection_pp']:.3f} & {r['attention_pp']:.3f} & {r['total_percent']:.3f}"
            for r in training['operation_contributions'] if r['scale'] == '14M']
    write('operation-summary', tabular('lrrr', r'Recipe & Projections (pp) & $QK+PV$ (pp) & Total $\Smodel$ (\%)', rows))
    rows = []
    for r in runtime['rows']:
        k = r['candidate_latency_ms']
        rows.append(f"{r['condition']} & {r['family']} & {'---' if r['dose'] is None else format(r['dose'],'g')} & {r['BF16_validation_loss']:.4f} & {r['paired_native_latency_ms']['k050']:.4f} & {k['k050-no-skip']:.4f} & {k['k050']:.4f} & {k['k050-attention-dense']:.4f} & {r['sparse_path_factor']:.4f}")
    write('runtime-latencies', tabular('llrrrrrrr', r'ID & Recipe & Dose & BF16 loss & Native & Skips off & K050 & Attn. dense & Factor', rows))
    rows=[]
    for key, fits in runtime['association_sensitivity'].items():
        label=key.replace('without_search_checkpoint','Without search checkpoint').replace('within_','Within ').replace('without_','Without ').replace('all','All checkpoints')
        rows.append(f"{label} & {fits['native_relative']['n']} & {fits['native_relative']['r_squared']:.4f} & {fits['sparse_path']['r_squared']:.4f}")
    write('runtime-sensitivity',tabular('lrrr',r'Checkpoint subset & $n$ & Native-relative $R^2$ & Sparse-factor $R^2$',rows))
    rows = []
    for r in runtime['diagnostics']:
        if r['condition'] not in ('c20', 'c30'):
            continue
        for h, z in zip(r['h_z_rows']['h']['layers'], r['h_z_rows']['z']['layers']):
            assert h['layer'] == z['layer'] and h['rows'] == z['rows'] == 338*2048
            rows.append(f"{r['family']} & {h['layer']} & {100*h['all_zero_row_fraction']:.2f} & {h['mean_nonzeros_per_row']:.3f} & {100*z['all_zero_row_fraction']:.2f} & {z['mean_nonzeros_per_row']:.3f}")
    write('projection-input-rows', tabular('lrrrrr', r'Recipe & Layer & $h=0$ rows (\%) & Mean NNZ $h$ & $z=0$ rows (\%) & Mean NNZ $z$', rows))
    sources = [HERE/'training-audit.json', HERE/'runtime-audit.json', PRIOR/'figure_data.json',
               *sorted((PRIOR/'manuscript-tables').glob('*.tex'))]
    (OUT/'SOURCES.json').write_text(json.dumps({str(p.relative_to(ROOT)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}, indent=2)+'\n')
    print('12 tables formatted; 54 endpoints, 29 pairs, 15 scale pairs and 30 matched latencies retained.')


if __name__ == '__main__':
    main()

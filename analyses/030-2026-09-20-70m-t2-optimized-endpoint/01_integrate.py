"""Add one verified endpoint without pooling the two timing sessions."""
import copy
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OLD = ROOT / 'analyses/028-2026-09-20-70m-optimized-grid'
RUN = ROOT / 'runs/047-2026-09-20-pythia70m-t2-kappa05-optimized-latency'
OBS = 'observations/001-later-70m-endpoint.md'
EVIDENCE = '% Source: Analysis030 observations/001-later-70m-endpoint.md; 01_integrate.py; Run047 results/matched-grid.json.'
MODES = ('native_graph', 'legacy_graph', 'candidate_graph')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2)+'\n', encoding='utf-8', newline='\n')


def tables(data, session):
    extra = next(r for r in data['trained_points'] if r.get('run047_id') == 'c26')
    # Keep every historical table cell; fill the already allocated endpoint row.
    text = (OLD/'tables/14m-70m.tex').read_text(encoding='utf-8')
    text = text.replace(text.splitlines()[0], EVIDENCE, 1)
    old_caption = text[text.index(r'\caption{'):text.index(r'\label{')]
    caption = (r'\caption{\textbf{14M/70M results.} OL1 pressure uses $\lambda=b=1$. '
               r'The original 26 70M checkpoints share one RTX\,5090 session (native Base: 1.611\,ms). '
               r'The $T_2/P_h$, $\kappa=0.5$ row uses a later session with the same frozen kernel and '
               r'protocol; its repeated references are given in Table~\ref{tab:70m-additional-endpoint}. '
               r'Absolute timings are not pooled or rescaled across sessions.}'+'\n')
    text = text.replace(old_caption, caption, 1)
    needle = '& --- & --- & --- & ---'
    assert text.count(needle) == 1
    values = [extra['loss'], extra['sparsity'], extra['implementation_latency_ms']['legacy_graph'],
              extra['implementation_latency_ms']['candidate_graph']]
    text = text.replace(needle, f'& {values[0]:.4f} & {values[1]:.3f} & {values[2]:.3f} & {values[3]:.3f}', 1)
    folder = HERE/'tables'
    folder.mkdir(exist_ok=True)
    (folder/'14m-70m.tex').write_text(text, encoding='utf-8', newline='\n')

    lines = [EVIDENCE, r'\begin{table}[!htbp]', r'\centering\small',
             r'\setlength{\tabcolsep}{5pt}',
             r'\caption{\textbf{Later 70M measurement and session references.} '
             r'The frozen optimized kernel is evaluated on the additional $T_2/P_h$, $\kappa=0.5$ '
             r'checkpoint, with Base and $\kappa=0.1$ repeated in the same later RTX\,5090 session. '
             r'Each row uses three fresh processes, 1,344 timings per implementation and full-validation '
             r'numerical checks. Earlier rows are retained for comparison; sessions are not pooled.}',
             r'\label{tab:70m-additional-endpoint}',
             r'\begin{tabular}{@{}llrrr@{}}', r'\toprule',
             r'Checkpoint & Session & Native (ms) & Port (ms) & Optimized (ms) \\', r'\midrule']
    labels = {'c00': r'Base', 'c25': r'$T_2/P_h$, $\kappa=0.1$', 'c26': r'$T_2/P_h$, $\kappa=0.5$'}
    for i, row in enumerate(session['conditions']):
        if i:
            lines.append(r'\midrule')
        cid = row['id']
        earlier = next((r for r in data['trained_points'] if r.get('run045_id') == cid), None)
        if earlier:
            lines.append(r'\multirow{2}{*}{'+labels[cid]+r'} & Earlier & '+
                         ' & '.join(f"{earlier['implementation_latency_ms'][m]:.3f}" for m in MODES)+r' \\')
        label = '' if earlier else labels[cid]
        lines.append(label+' & Later & '+' & '.join(f"{row['latency_ms'][m]:.3f}" for m in MODES)+r' \\')
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}', '']
    (folder/'70m-additional-endpoint.tex').write_text('\n'.join(lines), encoding='utf-8', newline='\n')


def main():
    receipt = read(RUN/'results/local-verification.json')
    session = read(RUN/'results/matched-grid.json')
    assert receipt['status'] == 'verified'
    assert session['status'] == 'complete' and session['all_qualified']
    assert session['checkpoints'] == 3 and session['fresh_processes'] == 9
    assert session['timings_per_implementation_checkpoint'] == 1344
    old = read(OLD/'data/full-trained-results.json')
    data = copy.deepcopy(old)
    row = next(r for r in data['trained_points'] if r.get('appendix_only'))
    assert row['checkpoint_key'] == 'd6a3813f83ce080b07637fe422bb7ab1b23a6793e39c365601301a661851d02f'
    inputs = read(RUN/'provenance/inputs.json')
    identity = next(r for r in inputs['checkpoints'] if r['id'] == 'c26')
    assert identity['final_checkpoint_content_sha256'] == row['checkpoint_key']
    # The complete frozen identity is retained as evidence, not inferred from the label.
    measured = next(r for r in session['conditions'] if r['id'] == 'c26')
    row['historical_run046_timing'] = {k: row[k] for k in (
        'latency_ms', 'kernel', 'timing_session', 'timing_device_uuid',
        'native_same_checkpoint_ms', 'paired_native_speedup', 'qualified')}
    row.update(run047_id='c26', appendix_only=False, timing_session='Run047', kernel='opt073',
               timing_device_uuid=session['device_uuid'],
               implementation_latency_ms=measured['latency_ms'], qualified=measured['qualified'],
               latency_ms=measured['latency_ms']['candidate_graph'],
               optimized_latency_ms=measured['latency_ms']['candidate_graph'],
               displayed_latency_ms=measured['latency_ms']['candidate_graph'],
               original_port_latency_ms=measured['latency_ms']['legacy_graph'],
               native_same_checkpoint_ms=measured['latency_ms']['native_graph'],
               paired_native_speedup=measured['paired_speedup']['native_over_candidate'],
               native_base_speedup=measured['native_base_speedup'],
               native_base_reference_ms=session['native_base_ms'],
               process_geomean_ranges_ms=measured['process_geomean_ranges_ms'],
               timing_note='Run047 later session; frozen opt073; no cross-session pooling or rescaling.')
    references = []
    for cid in ('c00', 'c25'):
        a = next(r for r in old['trained_points'] if r.get('run045_id') == cid)
        b = next(r for r in session['conditions'] if r['id'] == cid)
        references.append(dict(id=cid, earlier_ms=a['implementation_latency_ms'], later_ms=b['latency_ms'],
                               change_percent={m:100*(b['latency_ms'][m]/a['implementation_latency_ms'][m]-1) for m in MODES}))
    data['later_session'] = dict(run='Run047', native_base_ms=session['native_base_ms'],
                                 device_uuid=session['device_uuid'], references=references,
                                 added_checkpoint_key=row['checkpoint_key'], frozen_input_identity=identity)
    data['observation'] = OBS
    data['timing_note'] = '26 unchanged Run045 points plus one Run047 endpoint. No pooling or rescaling across sessions.'
    data['scope_note'] = 'All 27 trained 70M endpoints have optimized latency; sessions remain identified separately.'
    data['70m_figure_checkpoint_keys'] = [r['checkpoint_key'] for r in data['trained_points'] if r['model']=='70M']
    for path in (OLD/'data/full-trained-results.json', RUN/'results/matched-grid.json',
                 RUN/'results/local-verification.json', RUN/'provenance/inputs.json'):
        data['sources_sha256'][path.relative_to(ROOT).as_posix()] = sha(path)
    assert len(data['trained_points']) == 84
    for a, b in zip(old['trained_points'], data['trained_points']):
        if a.get('checkpoint_key') != row['checkpoint_key']:
            assert a == b
        else:
            for field in ('loss','sparsity','zero_product_count','model_product_count','source_attempt'):
                assert a[field] == b[field]
    assert data['clipping_points'] == old['clipping_points']
    write(HERE/'data/full-trained-results.json', data)
    figure = read(OLD/'data/70m-quality-sparsity-native-latency.json')
    figure.update(trained_points=[r for r in data['trained_points'] if r['model']=='70M'],
                  source_sha256=sha(HERE/'data/full-trained-results.json'), observation=OBS,
                  timing_note=data['timing_note'], later_session=data['later_session'])
    assert len(figure['trained_points']) == 27
    write(HERE/'data/70m-quality-sparsity-native-latency.json', figure)
    write(HERE/'data/session-comparison.json', dict(references=references, later=session))
    tables(data, session)
    print(json.dumps(dict(endpoint={k:row[k] for k in ('loss','sparsity','latency_ms','native_base_speedup')},
                          references=references), indent=2))


if __name__ == '__main__':
    main()

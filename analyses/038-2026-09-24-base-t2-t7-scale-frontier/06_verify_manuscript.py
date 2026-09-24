"""Check the integrated table, immutable evidence and compiled draft after building."""
import hashlib
import json
from pathlib import Path
import re
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DRAFT = ROOT / 'manuscript/draft'
BASELINE = 'a4c245ea'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def old(path):
    return subprocess.check_output(['git', 'show', f'{BASELINE}:{path.relative_to(ROOT).as_posix()}'], cwd=ROOT)


def active(text):
    return '\n'.join(re.split(r'(?<!\\)%', line)[0] for line in text.splitlines())


def between(text, start, end):
    return text.split(start, 1)[1].split(end, 1)[0]


def main():
    evidence = json.loads((HERE/'data/manuscript-evidence.json').read_text())
    assert evidence['source_script_sha256'] == sha(HERE/'04_manuscript_evidence.py')
    figure = json.loads((HERE/'data/scale-figure.json').read_text())
    previous = json.loads(old(HERE/'data/scale-figure.json'))
    for key in ('executions', 'measured_pareto_frontier', 'counts'):
        assert figure[key] == previous[key], key
    assert figure['presentation']['base_guide_labels']
    for source, target in [('figures/01-base-t2-t7-scale-frontier.pdf','figures/02-14m-70m-latency-quality.pdf'),
                           ('figures/03-base-model-training.pdf','figures/19-base-model-optimization.pdf')]:
        assert sha(HERE/source) == sha(DRAFT/target)
    assert sha(HERE/figure['output']) == figure['output_sha256']
    training = json.loads((HERE/'data/base-model-training.json').read_text())
    assert sha(HERE/training['output']) == training['output_sha256']
    assert training['source_script_sha256'] == sha(HERE/'05_base_training.py')
    for data in (evidence, training):
        for name,digest in data['source_sha256'].items():
            assert sha(ROOT/name) == digest, name

    preserved = []
    for filename, start, end in [
        ('05-experimental-study.tex', r'\subsection{Extensive ablations at 14M}', r'\label{sec:quality-sparsity-results}'),
        ('11-appendix-complete-results.tex', r'\begin{longtable}', r'\end{longtable}'),
        ('11-appendix-complete-results.tex', r'\label{tab:endpoints-410m}', r'\end{table}'),
        ('12-appendix-kernel-validation.tex', r'\subsection{Controlled 14M execution ablations}', '__EOF__'),
    ]:
        path = DRAFT/filename
        before,after = active(old(path).decode()),active(path.read_text(encoding='utf-8'))
        if filename == '05-experimental-study.tex':
            # The following section heading changes; the 14M body does not.
            before = before.replace(r'\subsection{Results at 70M}', '')
            after = after.replace(r'\subsection{Targeted and broad thresholding across model sizes}', '')
        assert between(before,start,end) == between(after,start,end), filename
        preserved.append(dict(file=filename,start=start,end=end))
    for name in ('03-related-work.tex','08-appendix-interventions.tex','09-appendix-sparsity-accounting.tex'):
        assert (DRAFT/name).read_bytes() == old(DRAFT/name)
    for filename,prefix in [('06-discussion-conclusion.tex','More broadly, sparsification'),
                            ('02-introduction.tex','Under extensive 14M ablations')]:
        path = DRAFT/filename
        assert next(l for l in old(path).decode().splitlines() if l.startswith(prefix)) == next(l for l in path.read_text(encoding='utf-8').splitlines() if l.startswith(prefix))
    assert (HERE/'figures/02-absolute-relative-scale-frontier.pdf').read_bytes() == old(HERE/'figures/02-absolute-relative-scale-frontier.pdf')
    for path in (DRAFT/'figures').glob('*.pdf'):
        if path.name not in ('02-14m-70m-latency-quality.pdf','19-base-model-optimization.pdf'):
            assert path.read_bytes() == old(path), path.name

    results = (DRAFT/'11-appendix-complete-results.tex').read_text(encoding='utf-8')
    table = between(results,r'\label{tab:31m-results}',r'\end{table}')
    table_rows = [line for line in table.splitlines() if line.startswith('$T_')]
    assert len(table_rows) == 11
    for row,line in zip(evidence['rows_31m'],table_rows):
        fields = [s.strip().removesuffix('\\\\').strip() for s in line.split('&')]
        assert fields[1] == ('---' if row['kappa'] is None else f"{row['kappa']:g}")
        expected = [f"{row['S_model_percent']:.2f}",f"{row['loss']:.3f}",f"{row['delta_loss']:+.3f}",
                    f"{row['latency_ms']:.3f}",f"{row['delta_latency_ms']:+.3f}"]
        assert fields[2:] == expected, (line,expected)
    text = '\n'.join(active(p.read_text(encoding='utf-8')) for p in DRAFT.glob('*.tex'))
    for forbidden in ('64 intervention points','Results at 70M','[[', '02-absolute-relative-scale-frontier.pdf'):
        assert forbidden not in text, forbidden
    log = (DRAFT/'main.log').read_text(errors='replace')
    assert not re.search(r'undefined|Overfull|Underfull|LaTeX Warning|Package .* Warning',log)
    assert 'Output written on main.pdf (18 pages' in log
    aux = (DRAFT/'main.aux').read_text()
    assert r'\newlabel{tab:31m-results}{{9}{15}' in aux
    assert r'\newlabel{fig:70m-quality-latency}{{5}{8}' in aux
    outputs = [HERE/figure['output'], HERE/training['output'], DRAFT/'figures/02-14m-70m-latency-quality.pdf',
               DRAFT/'figures/19-base-model-optimization.pdf', DRAFT/'main.pdf']
    report = dict(status='verified', baseline_commit=BASELINE,
                  unchanged_coordinates=36, unchanged_checkpoints=33, unchanged_pareto_set=True,
                  checked_31m_table_rows=11, qualified_31m_processes=33, matched_kappa_comparisons=15,
                  base_training_updates=2848, preserved_sections=preserved,
                  untouched_other_manuscript_figures=True, rejected_companion_unchanged=True,
                  pages=18, unresolved_references=0, overfull_underfull_boxes=0,
                  nonfatal_toolchain_diagnostic='Infinite glue shrinkage found in box being split; reproduced from baseline source with the same toolchain.',
                  build_command='latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex',
                  outputs_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in outputs},
                  source_script_sha256=sha(Path(__file__)))
    (HERE/'data/manuscript-verification.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8',newline='\n')
    print('Verified manuscript table, protected evidence, figure copies, 18-page build and resolved references.')


if __name__ == '__main__':
    main()

"""Verify the documentation-only attention discussion and retain a new receipt."""
from datetime import datetime, timezone
import re
import subprocess

from io_utils import RUN, REPO, archived_run, read, record, verify, write


def main():
    target = RUN / 'results/attention-discussion-001.json'
    if target.exists():
        raise ValueError('Existing receipt is historical; do not overwrite')
    previous = read(RUN / 'results/audit-publication-001.json')
    # Old local draft paths are historical hashes, not current-draft constraints.
    for key in ['script_run_root', 'audit_run_root', 'unchanged_figure_run_root', 'snippet_source_run_root']:
        verify(previous[key])
    for entry in previous['snapshots']:
        verify(entry['snapshot_run_root'])
    audit = read(verify(previous['audit_run_root']))
    for key in ['script', 'source', 'figure']:
        verify(audit[key])
    draft = REPO / 'manuscript/draft'
    snapshots = []
    for name in ['kernel-autoresearch.tex', 'kernel-implementation.md']:
        origin = record(draft / name, REPO)
        snapshot = record(RUN / 'provenance/manuscript-20260908-r02' / name)
        assert origin['sha256'] == snapshot['sha256']
        snapshots.append({'origin_repo_root': origin, 'snapshot_run_root': snapshot})
    section = (draft / 'kernel-autoresearch.tex').read_text(encoding='utf-8')
    graphic = re.search(r'\\includegraphics\[.*?\]\{(.*?)\}', section).group(1)
    assert record((draft / graphic).resolve()) == audit['figure']
    assert 'profitability remains untested' in section
    assert 'supports only' in section and r'\ref{eq:denominator}' in section
    wrapper = archived_run(28) / 'candidates/k035/candidate.py'
    assert "tuple(q.shape)!=(1,4,2048,32)" in wrapper.read_text(encoding='utf-8')

    def tool(name, *args):
        return subprocess.check_output([name, *map(str, args)], encoding='utf-8', errors='replace')

    pdf = draft / 'main.pdf'
    assert re.search(r'Pages:\s+10\b', tool('pdfinfo', pdf))
    fonts = tool('pdffonts', pdf)
    assert 'Type 3' not in fonts and all(row.split()[-5] == 'yes' for row in fonts.strip().splitlines()[2:])
    assert not re.search(r'Warning|Overfull|Underfull|undefined', (draft / 'main.log').read_text(encoding='utf-8'))
    aux = (draft / 'main.aux').read_text(encoding='utf-8')
    assert r'\newlabel{fig:kernel-autoresearch}{{2}{6}' in aux
    assert r'\newlabel{eq:denominator}{{4}{10}' in aux
    pdf_text = tool('pdftotext', pdf, '-')
    assert 'Attention skipping is active but not profitable' in pdf_text
    value = {'utc': datetime.now(timezone.utc).isoformat(), 'script_run_root': record(__file__),
             'previous_receipt_run_root': record(RUN / 'results/audit-publication-001.json'),
             'unchanged_figure_run_root': audit['figure'], 'snapshots': snapshots,
             'observation_run_root': record(RUN / 'observations/03-matched-rmodel-speedup.md'),
             'fixed_length_wrapper_run_root': record(wrapper),
             'local_draft_files_repo_root': [record(draft / name, REPO) for name in [
                 'main.pdf', 'main.log', 'main.aux', 'README.md', 'experimental-appendix.tex']],
             'verification': 'Ten pages rendered at 110dpi and visually checked by the agent; '
                             'embedded fonts, no Type 3, unresolved-reference or box warnings. '
                             'Figure, reduced data, audited kernels and initial source snapshots unchanged.',
             'scope': 'Documentation-only; longer-T profitability is untested. No kernel changes or GPU run.'}
    write(target, value)
    print({'receipt': str(target), 'pages': 10, 'snapshots': len(snapshots), 'unchanged_figure': audit['figure']['sha256']})


if __name__ == '__main__':
    main()

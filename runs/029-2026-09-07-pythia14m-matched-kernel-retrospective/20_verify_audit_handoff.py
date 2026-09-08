"""Verify the approved manuscript addition; write a new, run-owned handoff receipt."""
from datetime import datetime, timezone
import re
import subprocess
import textwrap
import xml.etree.ElementTree as ET

from io_utils import RUN, REPO, archived_run, fs, read, record, verify, write


def main():
    destination = RUN / 'results/audit-publication-001.json'
    if destination.exists():
        raise ValueError('Handoff receipt already exists; do not overwrite')
    audit_path = RUN / 'results/implementation-audit-001.json'
    audit = read(audit_path)
    for key in ['script', 'source', 'figure']:
        verify(audit[key])
    verify(audit['agent_identity_source'], REPO)
    assert audit['timing_pairs'] == 522816 and audit['comparison_count'] == 391

    draft = REPO / 'manuscript/draft'
    snapshots = []
    for name in ['kernel-autoresearch.tex', 'kernel-implementation.md']:
        original = record(draft / name, REPO)
        snapshot = record(RUN / 'provenance/manuscript-20260908' / name)
        assert original['sha256'] == snapshot['sha256']
        snapshots.append({'origin_repo_root': original, 'snapshot_run_root': snapshot})
    section = (draft / 'kernel-autoresearch.tex').read_text(encoding='utf-8')
    note = (draft / 'kernel-implementation.md').read_text(encoding='utf-8')
    figure_path = re.search(r'\\includegraphics\[.*?\]\{(.*?)\}', section).group(1)
    assert record((draft / figure_path).resolve()) == audit['figure']
    assert r'\input{kernel-autoresearch}' in (draft / 'experimental-study.tex').read_text(encoding='utf-8')
    for target in re.findall(r'\[[^\]]*\]\(([^)]+)\)', note):
        assert fs((draft / target).resolve()).is_file(), target
    snippet = re.search(r'```cpp\n(.*?)\n```', note, re.S).group(1)
    primitive = archived_run(28) / 'candidates/k042/projection.cu'
    source = fs(primitive).read_text(encoding='utf-8')
    match = re.search(r'    CUTLASS_DEVICE void operator\(\).*?\n    }', source, re.S)
    assert match is not None and snippet == textwrap.dedent(match.group(0))

    tests = []
    for name, total, skipped in [('audit-handoff-tests-001.xml', 320, 1),
                                 ('audit-frozen-components-001.xml', 8, 0),
                                 ('audit-counterexamples-001.xml', 4, 0)]:
        path = RUN / 'runtime' / name
        suite = ET.parse(path).getroot().find('testsuite')
        counts = {key: int(suite.get(key)) for key in ['tests', 'errors', 'failures', 'skipped']}
        assert counts == {'tests': total, 'errors': 0, 'failures': 0, 'skipped': skipped}
        tests.append({'receipt_run_root': record(path), 'counts': counts,
                      'passed': total - skipped, 'seconds': float(suite.get('time'))})

    def pdf_tool(name, *args):
        return subprocess.check_output([name, *map(str, args)], encoding='utf-8', errors='replace')

    pdf = draft / 'main.pdf'
    info = pdf_tool('pdfinfo', pdf)
    assert re.search(r'Pages:\s+10\b', info)
    fonts = pdf_tool('pdffonts', pdf)
    font_rows = fonts.strip().splitlines()[2:]
    assert font_rows and 'Type 3' not in fonts
    assert all(row.split()[-5] == 'yes' for row in font_rows)
    log = (draft / 'main.log').read_text(encoding='utf-8')
    assert not re.search(r'Warning|Overfull|Underfull|undefined', log)
    aux = (draft / 'main.aux').read_text(encoding='utf-8')
    assert r'\newlabel{sec:kernel-autoresearch}{{4.1}{5}' in aux
    assert r'\newlabel{fig:kernel-autoresearch}{{2}{6}' in aux
    cited = {key for group in re.findall(r'\\citation\{(.*?)\}', aux) for key in group.split(',')}
    resolved = set(re.findall(r'\\bibcite\{(.*?)\}', aux))
    assert cited <= resolved
    pdf_text = pdf_tool('pdftotext', pdf, '-')
    for phrase in ['Realizing sparsity with specialized inference kernels', '0.781',
                   'Matched kernel auto-research and sparsity-dependent acceleration']:
        assert phrase in pdf_text

    value = {
        'utc': datetime.now(timezone.utc).isoformat(), 'script_run_root': record(__file__),
        'audit_run_root': record(audit_path), 'unchanged_figure_run_root': audit['figure'],
        'snapshots': snapshots, 'snippet_source_run_root': record(primitive), 'tests': tests,
        'test_source_run_root': record(RUN / 'tests/test_audit.py'),
        'local_draft_files_repo_root': [record(draft / name, REPO) for name in [
            'main.tex', 'experimental-study.tex', 'README.md', 'experimental-notes.md',
            'kernel-autoresearch.tex', 'kernel-implementation.md', 'main.pdf', 'main.log', 'main.aux']],
        'pdf_checks': {'pages': 10, 'embedded_fonts': len(font_rows), 'type3_fonts': 0,
                       'unresolved_citations': sorted(cited - resolved), 'resolved_citation_keys': len(cited),
                       'latex_warnings_or_box_errors': False, 'section_page': 5, 'figure_page': 6},
        'visual_review': 'Agent inspected all 10 pages rendered at 110 dpi from this exact PDF; '
                         'no clipping, overlaps or unreadable symbols. Reading wrapper, not ICLR-template fit.',
        'disposition': 'Scoped agent-assisted systems case and positive association supported; '
                       'no universal monotonic or equal-quality claim. Non-cohort gate bug documented, '
                       'frozen kernels and figure unchanged. No new GPU execution.'}
    write(destination, value)
    print({'receipt': str(destination), 'pdf_checks': value['pdf_checks'],
           'test_pass_counts': [entry['passed'] for entry in tests], 'snapshots': len(snapshots)})


if __name__ == '__main__':
    main()

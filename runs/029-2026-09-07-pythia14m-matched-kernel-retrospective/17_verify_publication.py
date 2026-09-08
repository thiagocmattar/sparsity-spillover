"""Verify final evidence references, test report and current vector PDF assets."""
import subprocess
import re
import xml.etree.ElementTree as ET
from io_utils import RUN, read, write, verify, record


def main():
    data_path = RUN/'results/matched-retrospective-001.json'
    data = read(data_path)
    for row in data['sources']:
        verify(row)
    plan = read(verify(data['plan']))
    completed = read(RUN/'artifacts/scientific/completed.json')
    assert {r['job']['key'] for r in completed} == {r['key'] for r in plan['jobs']}
    assert len(completed) == len(plan['jobs'])
    report_path = RUN/'results/report-001.json'
    report = read(report_path)
    verify(report['source']); verify(report['script'])
    assert sum(report['process_outcomes'].values()) == len(completed)
    inputs = read(RUN/'provenance/inputs.json')
    assert {d['condition'] for d in data['diagnostics']} == {c['id'] for c in inputs['checkpoints']}
    for row in read(RUN/'provenance/archive.json')['files']:
        verify(row['snapshot'])
    for row in [inputs['validation']]+[f for c in inputs['checkpoints'] for f in c['files']+c['provenance']]:
        verify(row)
    manifest_path = RUN/'results/figures-002.json'
    manifest = read(manifest_path)
    verify(manifest['source']); verify(manifest['script'])
    figures = []
    def output(command):
        return subprocess.run(command, capture_output=True, text=True, check=True).stdout
    for row in manifest['figures']:
        path = verify(row)
        info = output(['pdfinfo', str(path)])
        assert int(re.search(r'^Pages:\s+(\d+)', info, re.M).group(1)) == 1
        size = re.search(r'^Page size:\s+([\d.]+) x ([\d.]+)', info, re.M)
        images = output(['pdfimages', '-list', str(path)])
        assert not any(line.strip() for line in images.splitlines()[2:]), 'Publication figure must remain vector-only'
        assert 'Full-model speedup' in output(['pdftotext', '-layout', str(path), '-'])
        fonts = output(['pdffonts', str(path)])
        font_rows = [line.split() for line in fonts.splitlines()[2:] if line.strip()]
        assert font_rows and all(row[-5:-2] == ['yes', 'yes', 'yes'] for row in font_rows)
        assert all('TrueType' in row for row in font_rows)
        figures.append({'file': row, 'pages': 1, 'raster_images': 0,
                        'page_points': [float(size.group(1)), float(size.group(2))],
                        'embedded_subset_unicode_fonts': True, 'font_report': fonts})
    test_path = RUN/'runtime/final-tests.xml'
    suites = ET.parse(test_path).getroot().findall('testsuite')
    totals = {key: sum(int(s.get(key, 0)) for s in suites) for key in ['tests', 'failures', 'errors', 'skipped']}
    assert totals['failures'] == totals['errors'] == 0
    totals['passed'] = totals['tests']-totals['skipped']
    value = {'matrix_processes': len(completed), 'comparisons': len(data['points']),
             'final_diagnostics': len(data['diagnostics']), 'test_summary': totals,
             'evidence_hashes_verified': True, 'source_and_input_hashes_verified': True,
             'figures': figures, 'tests': record(test_path), 'data': record(data_path),
             'report': record(report_path), 'publication_manifest': record(manifest_path),
             'script': record(__file__)}
    write(RUN/'results/verification-001.json', value)
    print({k: value[k] for k in ['matrix_processes', 'comparisons', 'final_diagnostics', 'test_summary']})


if __name__ == '__main__':
    main()

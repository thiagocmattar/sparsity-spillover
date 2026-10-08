"""Verify actual analysis outputs, original identities, links and PDF bounds."""
import csv
import hashlib
import json
from pathlib import Path
import re

import fitz
import labels

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def identity(row):
    path = REPO / row['path']
    assert path.stat().st_size == row['bytes'], path
    assert hashlib.sha256(path.read_bytes()).hexdigest() == row['sha256'], path


def main():
    sources = read(ROOT / 'provenance/sources.json')
    generated = read(ROOT / 'provenance/generated.json')
    for row in sources['sources'] + generated['scripts'] + [generated['input']] + generated['outputs']:
        identity(row)
    data = read(ROOT / 'results/decomposition.json')
    assert data['status'] == 'verified'
    assert len(data['conditions']) == 6 and data['total_logit_comparisons'] == 44616
    assert data['semantic_styles'] == labels.COMMON_STYLES
    assert data['semantic_category_order'] == list(labels.COMMON_STYLES)
    original_sources = {}
    for c in data['conditions']:
        run = c['source_run']
        if run not in original_sources:
            original_sources[run] = read(REPO / 'runs' / run / 'results/decomposition.json')
        original = original_sources[run]
        pair = c['kappa'].replace('0.1', '0p1')
        condition = original['pairs'][pair] if c['model'] == '14M' else original['conditions'][c['key'].lower()]
        assert c['tiled_threshold'] == (c['model'] == '14M')
        assert abs(c['native_loss'] - c['sparse_loss']) < .001
        for name, m in c['modes'].items():
            previous = condition['modes'][name]
            assert m['host'] == previous['baseline_host_us']
            assert m['cuda_outer'] == previous['baseline_cuda_us']
            for key, original_key in [('active_us', 'mean_active_us'), ('span_us', 'mean_span_us'), ('gap_us', 'mean_gap_us')]:
                assert m[key] == previous['trace_budget'][original_key]
            assert m['host']['count'] == 1344 and len(m['process_host_geomean_us']) == 3
            assert abs(sum(m['categories'].values()) - m['active_us']) < .1
            assert list(m['semantic_categories']) == list(labels.COMMON_STYLES)
            assert abs(sum(m['semantic_categories'].values()) - m['active_us']) < .1
            assert abs(m['semantic_categories']['H/Z output block'] - sum(m['branches_by_layer'].values())) < .001
            assert abs(sum(labels.detail_values(m['categories']).values()) - m['semantic_categories']['H/Z output block']) < .001
            assert abs(m['active_us'] + m['gap_us'] - m['span_us']) < .001
    with (ROOT / 'results/forward-stages.csv').open(encoding='utf-8', newline='') as handle:
        stages = list(csv.DictReader(handle))
    assert len(stages) == 1286
    assert all((r['tiled_threshold'] == 'True') == (r['model'] == '14M') for r in stages)
    with (ROOT / 'results/semantic-categories.csv').open(encoding='utf-8', newline='') as handle:
        common = list(csv.DictReader(handle))
    assert len(common) == 198
    assert all(r['color'] == labels.COMMON_STYLES[r['category']]['color'] for r in common)
    broken = []
    for path in [ROOT / 'README.md', ROOT / 'LABELS.md'] + list((ROOT / 'observations').glob('*.md')):
        for target in re.findall(r'\]\(([^)]+)\)', path.read_text(encoding='utf-8')):
            if target.startswith(('https://', 'http://', '#')):
                continue
            if not (path.parent / target.split('#', 1)[0]).exists():
                broken.append({'file': str(path), 'link': target})
    assert not broken, broken
    qa = REPO / 'tmp/analysis042-figure-qa'
    qa.mkdir(parents=True, exist_ok=True)
    pages, bad = 0, []
    pdfs = sorted((ROOT / 'figures').glob('*.pdf'))
    assert len(pdfs) == 8
    for path in pdfs:
        with fitz.open(path) as doc:
            assert len(doc) == 1
            page = doc[0]
            text = page.get_text()
            if '14m' in path.name:
                assert 'TILED' in text and '8 x 16' in text
            elif any(s in path.name for s in ['31m', '70m']):
                assert 'SCALAR' in text and 'not tiled' in text
            else:
                assert 'TILED' in text and 'SCALAR' in text
            if 'decomposition' in path.name or 'dense-sparse' in path.name:
                normalized = ' '.join(text.split())
                assert all(category in normalized for category in labels.COMMON_STYLES), path
                assert 'Joint H/Z gates' not in normalized, path
            if 'hz-detail' in path.name:
                normalized = ' '.join(text.split())
                assert all(category in normalized for category in labels.DETAIL_HATCHES), path
            page.get_pixmap(matrix=fitz.Matrix(1.2, 1.2)).save(qa / (path.stem + '.png'))
            for block in page.get_text('dict')['blocks']:
                for line in block.get('lines', []):
                    for span in line['spans']:
                        if not page.rect.contains(fitz.Rect(span['bbox'])):
                            bad.append({'pdf': path.name, 'text': span['text'], 'bbox': span['bbox']})
            pages += 1
    assert not bad, bad
    record = {'status': 'verified', 'source_and_output_hashes_match': True,
              'forward_stage_rows': len(stages), 'all_relative_markdown_links_exist': True,
              'pdf_files': len(pdfs), 'pdf_pages': pages, 'out_of_page_text': bad,
              'tiled_scalar_labels_verified': True, 'publication_format': 'PDF only',
              'common_category_rows': len(common), 'fixed_semantic_colors_verified': True,
              'common_active_totals_and_hz_child_sums_match': True,
              'original_mode_budgets_preserved': True,
              'temporary_renders': 'Ignored tmp/analysis042-figure-qa/',
              'visual_review': 'Pending separate inspection of the rendered pages.'}
    (ROOT / 'results/verification.json').write_bytes((json.dumps(record, indent=2) + '\n').encode())
    print('Verified hashes, 1286 original stage rows, 198 common-category rows, H/Z sums, fixed colors, links and 8 PDF bounds.')


if __name__ == '__main__':
    main()

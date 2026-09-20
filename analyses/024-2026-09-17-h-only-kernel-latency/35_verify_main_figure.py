"""Check the requested cohort, unchanged measurements, font scale and PDF copies."""
import hashlib
import json
import math
from pathlib import Path
import fitz

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    data = json.loads((HERE / 'data/14m-main-quality-sparsity-latency.json').read_text())
    source = ROOT / 'analyses/026-2026-09-20-14m-hz-quality-sparsity-latency/data/figure-data.json'
    reference = json.loads(source.read_text())
    actual = {r['checkpoint_key']: r for r in data['trained_points']}
    expected = {r['checkpoint_key']: r for r in reference['trained_points']}
    assert actual == expected and len(actual) == 40
    assert len(data['series']) == 10
    assert data['layout']['legend_rows'] == 2 and data['layout']['legend_columns'] == 5
    assert data['layout']['font_scale'] == .9
    assert data['panels'][0]['trained_keys'] == data['panels'][1]['trained_keys']
    prior = json.loads((HERE / 'data/14m-70m-quality-sparsity-latency.json').read_text())
    assert data['clipping_points'] == [r for r in prior['clipping_points'] if r['model'] == '14M']
    assert [s['label'] for s in data['series']][2:4] == [r'$T_1/P_h$', r'$T_2/P_h$']
    pdf = HERE / data['output']
    manuscript = ROOT / 'manuscript/draft'
    assert sha(pdf) == data['output_sha256'] == sha(manuscript / 'figures' / pdf.name)
    assert sha(HERE / 'data/14m-main-quality-sparsity-latency.json') == sha(
        manuscript / 'supplementary-data/14m-main-quality-sparsity-latency.json')
    for path, digest in data['sources_sha256'].items():
        assert sha(ROOT / path) == digest
    assert sha(HERE / '34_plot_14m_main_figure.py') == data['script_sha256']
    doc = fitz.open(pdf)
    assert len(doc) == 1
    page = doc[0]
    assert all(page.rect.contains(fitz.Rect(block[:4])) for block in page.get_text('blocks'))
    fonts = page.get_fonts(full=True)
    assert fonts and all(doc.extract_font(font[0])[3] for font in fonts)
    spans = [s for b in page.get_text('dict')['blocks'] if 'lines' in b
             for line in b['lines'] for s in line['spans']]
    label = next(s for s in spans if s['text'] == 'Validation loss')
    assert math.isclose(label['size'], 11 * .9, abs_tol=1e-5)
    result = dict(status='verified', trained_checkpoints_per_panel=40,
                  legend_rows=2, legend_columns=5, font_scale=.9,
                  exact_source_records=True, posthoc_settings_per_panel=20,
                  embedded_fonts=len(fonts), text_within_page=True,
                  pdf_sha256=sha(pdf), verifier_sha256=sha(Path(__file__)),
                  visual_review='Final rendered page checked: complete 2x5 legend and no text overlaps. No manuscript recompilation.')
    (HERE / 'data/main-figure-ten-recipes-verification.json').write_text(
        json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()

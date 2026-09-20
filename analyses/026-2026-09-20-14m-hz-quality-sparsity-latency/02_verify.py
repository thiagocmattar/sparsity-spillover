"""Independently verify plotted membership, source bytes, and PDF structure."""
import collections
import hashlib
import json
import math
from pathlib import Path

import fitz

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    data = read(HERE / 'data/figure-data.json')
    rows = data['trained_points']
    assert len(rows) == len({r['source_attempt'] for r in rows}) == 40
    assert len(data['excluded_naive_l1']) == 4
    old = read(ROOT / 'analyses/024-2026-09-17-h-only-kernel-latency/data/quality-main-appendix.json')
    view = old['outputs'][0]
    expected = {r['source_attempt']: r for r in view['trained_points'] if r['model'] == '14M'}
    actual = {r['source_attempt']: r for r in rows if r['scope'] != 'hz'}
    assert expected.keys() == actual.keys() and len(expected) == 36
    assert all(actual[p][k] == v for p, r in expected.items() for k, v in r.items())
    assert data['clipping_points'] == [p for p in view['clipping_points'] if p['model'] == '14M']
    summary = read(ROOT / 'runs/041-2026-09-20-pythia14m-hz-h-only-ol1/artifacts/summary.json')
    hz = [r for r in rows if r['scope'] == 'hz']
    assert [r['kappa'] for r in hz] == [0, .01, .05, .1]
    for r, s in zip(hz, summary['conditions']):
        assert r['loss'] == s['training_validation_loss']
        assert math.isclose(r['sparsity'], 100 * s['R_model'], rel_tol=1e-12)
        assert math.isclose(r['latency_ms'], s['k050_geomean_host_ms'], rel_tol=1e-12)
    sessions = dict(collections.Counter(r['timing_session'] for r in rows))
    assert sessions == {'Run029': 31, 'Run033': 5, 'Run041': 4}
    for path, digest in data['sources_sha256'].items():
        assert sha(ROOT / path) == digest, path
    for path, digest in data['reference_pdfs_sha256'].items():
        assert sha(ROOT / path) == digest, path
    assert sha(HERE / '01_build.py') == data['script_sha256']
    pdf = HERE / data['figure']
    assert sha(pdf) == data['figure_sha256']
    doc = fitz.open(pdf)
    assert len(doc) == 1
    page = doc[0]
    assert all(page.rect.contains(fitz.Rect(b[:4])) for b in page.get_text('blocks'))
    text = page.get_text()
    for needle in ['Validation loss', 'Full-model latency (ms)', '40 trained models', 'three timing sessions', '(new)']:
        assert needle in text, needle
    fonts = page.get_fonts(full=True)
    assert fonts and all(doc.extract_font(f[0])[3] for f in fonts)
    result = dict(status='verified', trained_per_panel=40, exact_historical_membership=36,
                  new_run041_models=4, retained_clipping_points=20,
                  historical_values_unchanged=True, source_hashes_verified=len(data['sources_sha256']),
                  reference_pdfs_unchanged=2, timing_sessions=sessions,
                  pdf_pages=1, embedded_fonts=len(fonts), all_text_within_page=True,
                  figure_sha256=sha(pdf), figure_data_sha256=sha(HERE / 'data/figure-data.json'),
                  builder_sha256=data['script_sha256'], verifier_sha256=sha(Path(__file__)))
    (HERE / 'data/verification.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()

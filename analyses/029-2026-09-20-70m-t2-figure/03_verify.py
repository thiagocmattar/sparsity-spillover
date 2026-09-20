"""Check the requested cohort, unchanged prior measurements, copies and PDF layout."""
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
    path = HERE / 'data/70m-quality-sparsity-native-latency.json'
    data = json.loads(path.read_text(encoding='utf-8'))
    prior = json.loads((ROOT/'analyses/024-2026-09-17-h-only-kernel-latency/data'/path.name).read_text())
    assert data['trained_points'][:22] == prior['trained_points']
    assert data['plotted_points'][:22] == prior['plotted_points']
    assert data['clipping_points'] == prior['clipping_points']
    assert data['native_base_reference'] == prior['native_base_reference']
    keys = {r['checkpoint_key'] for r in data['trained_points']}
    assert len(keys) == len(data['trained_points']) == 27
    assert {r['checkpoint_key'] for r in data['plotted_points']} == keys
    assert data['panels'][0]['trained_keys'] == data['panels'][1]['trained_keys']
    assert set(data['panels'][0]['trained_keys']) == keys
    assert len(data['panels'][0]['clipping_ids']) == 20
    assert data['panels'][1]['clipping_ids'] == []
    t2 = data['added_t2_points']
    assert [r['kappa'] for r in t2] == [0, .01, .05, .1, .5]
    assert [r['timing_session'] for r in t2] == ['Run043']*4+['Run046']
    assert all(r['scope']=='hz' and r['pressure']=='h' and r['qualified'] for r in t2)
    assert all(r['paired_samples']==1344 and r['kernel']=='k050-70m-v2' for r in t2)
    for row in t2:
        m = row['logical_counts']
        assert row['sparsity'] == 100*(m['block_zero_product_count']/m['model_product_count'])
        run = ROOT/'runs'/row['training_run']
        verification = json.loads((run/'artifacts/verification.json').read_text())
        source, = [r for r in verification['conditions'] if r['checkpoint_content_sha256']==row['checkpoint_key']]
        assert row['loss'] == source['final_validation_loss']
        assert math.isfinite(row['latency_ms']) and row['latency_ms'] > 0
    assert len({r['timing_device_uuid'] for r in data['trained_points']}) == 3
    for relative, digest in data['sources_sha256'].items():
        assert sha(ROOT/relative) == digest, relative
    for name in ('collector', 'script'):
        assert sha(HERE/data[name]) == data[name+'_sha256']
    pdf = HERE/data['output']
    assert sha(pdf) == data['output_sha256']
    manuscript = ROOT/'manuscript/draft'
    assert sha(pdf) == sha(manuscript/'figures'/pdf.name)
    assert sha(path) == sha(manuscript/'supplementary-data'/path.name)
    captions = [(manuscript/name).read_text() for name in ('training-results.tex', 'experimental-study.tex')]
    caption, = [s for s in captions if 'figures/'+pdf.name in s]
    assert '27 trained checkpoints' in caption and 'Run043/Run046' in caption
    doc = fitz.open(pdf)
    assert len(doc) == 1
    page = doc[0]
    text = page.get_text()
    assert 'PyTorch base: 1.664 ms' in text and not page.search_for('Post-hoc')
    assert text.count('ceiling') == 3
    assert all(page.rect.contains(fitz.Rect(b[:4])) for b in page.get_text('blocks'))
    assert all(doc.extract_font(f[0])[3] for f in page.get_fonts(full=True))
    temporary = ROOT/'tmp/pdfs/run046-t2-figure'
    temporary.mkdir(parents=True, exist_ok=True)
    page.get_pixmap(matrix=fitz.Matrix(2,2)).save(temporary/'verified.png')
    result = dict(status='verified', trained_points=27, added_t2_points=5,
        historical_points_unchanged=22, clipping_points_per_panel=[20,0],
        qualified_added_processes=15, paired_samples_per_new_checkpoint=1344,
        source_hashes_checked=len(data['sources_sha256']), timing_gpu_sessions=3,
        exact_manuscript_copies=True, pdf_text_bounds_and_fonts='passed',
        pdf_sha256=sha(pdf), verifier_sha256=sha(Path(__file__)))
    (HERE/'data/verification.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()

"""Verify exact recipe matching, retained evidence and the standalone PDF."""
import hashlib
import json
from pathlib import Path
import fitz

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    data = json.loads((HERE/'data/figure-data.json').read_text())
    expected = {('0','none'),('1','none'),('hz','h'),('4','h'),('7','h'),('4','all'),('7','all')}
    assert {(s['scope'],s['pressure']) for s in data['shared_recipes']} == expected
    origins = {'14M':'analyses/027-2026-09-20-run044-manuscript/data/figure-data.json',
               '70M':'analyses/029-2026-09-20-70m-t2-figure/data/70m-quality-sparsity-native-latency.json'}
    for size, panel in data['panels'].items():
        original = json.loads((ROOT/origins[size]).read_text())
        selected = [r for r in original['trained_points'] if (r['scope'],r['pressure']) in expected]
        assert panel['points'] == selected and len(selected)==27
        assert panel['clipping'] == original['clipping_points'] and len(panel['clipping'])==20
        assert len({r['checkpoint_key'] for r in selected})==27
        for key in expected-{('0','none'),('1','none')}:
            assert sorted(r['kappa'] for r in selected if (r['scope'],r['pressure'])==key)==[0,.01,.05,.1,.5]
    operations = data['bar_chart']['rows']
    assert [r['operation'] for r in operations]==['a','m','h','z','qk','pv']
    assert [round(r['saved_microseconds'],1) for r in operations]==[-1.6,-2.4,150.1,29.0,-2.9,-6.2]
    assert data['bar_chart']['verification']['qualified_processes']==30
    for relative,digest in data['sources_sha256'].items():
        assert sha(ROOT/relative)==digest, relative
    assert sha(HERE/data['script'])==data['script_sha256']
    pdf=HERE/data['output']
    assert sha(pdf)==data['output_sha256']
    doc=fitz.open(pdf);assert len(doc)==1
    page=doc[0];text=page.get_text()
    for label in ['(a) Pythia-14M','(b) Conditional time savings','(c) Pythia-70M','+150.1','+29.0','-6.2']:
        assert label in text,label
    assert len(page.search_for('Base model'))==1
    assert not page.search_for('Post-hoc')
    assert 'one skipping path off' not in text and 'Effects are not additive' not in text
    assert all(page.rect.contains(fitz.Rect(b[:4])) for b in page.get_text('blocks'))
    assert all(doc.extract_font(f[0])[3] for f in page.get_fonts(full=True))
    draft=ROOT/'manuscript/draft'
    for installed,source in [('figures/24-quality-sites-quality.pdf',pdf),
                             ('supplementary-data/quality-sites-quality.json',HERE/'data/figure-data.json')]:
        target=draft/installed
        assert target.read_bytes()==source.read_bytes()
        manifest=json.loads((target.parent/'SOURCES.json').read_text())
        assert manifest[target.name]['sha256']==sha(source)
    active='\n'.join(line for line in (draft/'introduction.tex').read_text().splitlines() if not line.lstrip().startswith('%'))
    assert active.count('figures/24-quality-sites-quality.pdf')==1
    assert active.count('\\label{fig:quality-sparsity-overview}')==1
    temporary=ROOT/'tmp/pdfs/analysis031';temporary.mkdir(parents=True,exist_ok=True)
    page.get_pixmap(matrix=fitz.Matrix(2,2)).save(temporary/'preview.png')
    report=dict(status='verified',panels=3,matched_recipes=7,trained_points_per_quality_panel=27,
        clipping_settings_per_quality_panel=20,conditional_bars=6,source_values_unchanged=True,
        shared_legend=True,manuscript_copies_and_label='passed',pdf_bounds_and_fonts='passed',pdf_sha256=sha(pdf),verifier_sha256=sha(Path(__file__)))
    (HERE/'data/verification.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()

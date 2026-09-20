"""Install the approved additions and verify coordinates, tables and provenance."""
import hashlib
import json
import math
from pathlib import Path
import shutil

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DRAFT = ROOT/'manuscript/draft'
OLD = ROOT/'analyses/028-2026-09-20-70m-optimized-grid'
RUN = ROOT/'runs/047-2026-09-20-pythia70m-t2-kappa05-optimized-latency'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    data = read(HERE/'data/full-trained-results.json')
    old = read(OLD/'data/full-trained-results.json')
    raw = read(RUN/'results/matched-grid.json')
    rows = data['trained_points']
    extra, = [r for r in rows if r.get('run047_id') == 'c26']
    original_by_source = {r['source_attempt']:r for r in old['trained_points']}
    assert len(rows) == len({r['source_attempt'] for r in rows}) == 84
    assert {size:sum(r['model']==size for r in rows) for size in ('14M','70M','410M')} == {'14M':45,'70M':27,'410M':12}
    for row in rows:
        original = original_by_source[row['source_attempt']]
        if row is not extra:
            assert row == original
        assert math.isclose(row['sparsity'],100*row['zero_product_count']/row['model_product_count'],rel_tol=1e-12)
    assert data['clipping_points'] == old['clipping_points']
    assert data['paired_pressure'] == old['paired_pressure']
    assert data['main_figure_checkpoint_keys'] == old['main_figure_checkpoint_keys']
    figure = read(HERE/'data/70m-quality-sparsity-native-latency.json')
    plot = read(HERE/'data/23-70m-quality-sparsity-native-latency.json')['evidence']
    assert len(figure['trained_points']) == len(plot['checkpoint_keys']) == 27
    assert set(plot['checkpoint_keys']) == {r['checkpoint_key'] for r in figure['trained_points']}
    assert plot['native_base_ms'] == old['matched_summary']['native_base_ms']
    assert plot['later_endpoint_key'] == extra['checkpoint_key']
    measured = next(r for r in raw['conditions'] if r['id']=='c26')
    assert extra['implementation_latency_ms'] == measured['latency_ms']
    assert extra['displayed_latency_ms'] == measured['latency_ms']['candidate_graph']
    assert extra['native_base_reference_ms'] == raw['native_base_ms']
    assert extra['native_base_speedup'] == measured['native_base_speedup']
    table = (HERE/'tables/14m-70m.tex').read_text(encoding='utf-8')
    for row in (r for r in rows if r['model']=='70M'):
        fields = [f"{row['loss']:.4f}",f"{row['sparsity']:.3f}"]
        fields += [f"{row['implementation_latency_ms'][m]:.3f}" for m in ('legacy_graph','candidate_graph')]
        assert ' & '.join(fields) in table
    references = (HERE/'tables/70m-additional-endpoint.tex').read_text(encoding='utf-8')
    for row in raw['conditions']:
        assert ' & '.join(f"{row['latency_ms'][m]:.3f}" for m in ('native_graph','legacy_graph','candidate_graph')) in references
    for name, digest in data['sources_sha256'].items():
        assert sha(ROOT/name) == digest, name
    copied = {}
    for folder, names, script in [
        ('figures', {'23-70m-quality-sparsity-native-latency.pdf':'23-70m-quality-sparsity-native-latency.pdf'}, '02_plot.py'),
        ('supplementary-data', {'all-model-quality-sparsity.json':'full-trained-results.json',
          '70m-quality-sparsity-native-latency.json':'70m-quality-sparsity-native-latency.json',
          'kernel/70m-later-session.json':'session-comparison.json'}, '01_integrate.py')]:
        manifest = DRAFT/folder/'SOURCES.json'
        sources = read(manifest)
        for target_name, source_name in names.items():
            source = HERE/('figures' if folder=='figures' else 'data')/source_name
            target = DRAFT/folder/target_name
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(source,target)
            assert sha(source)==sha(target)
            record = dict(source=source.relative_to(ROOT).as_posix(),sha256=sha(source),bytes=source.stat().st_size,
                          observation=(HERE/'observations/001-later-70m-endpoint.md').relative_to(ROOT).as_posix(),
                          script=(HERE/script).relative_to(ROOT).as_posix())
            sources[target_name]=record
            copied[target.relative_to(DRAFT).as_posix()]=record['sha256']
        manifest.write_text(json.dumps(sources,indent=2)+'\n',encoding='utf-8',newline='\n')
    for name in ('14m-70m.tex','70m-additional-endpoint.tex'):
        source=HERE/'tables'/name
        target=DRAFT/'tables/compact-results'/name
        shutil.copy2(source,target)
        assert sha(source)==sha(target)
        copied[target.relative_to(DRAFT).as_posix()]=sha(target)
    result = dict(status='verified',trained_conditions=84,figure_70m_points=27,
                  original_run045_rows_unchanged=26,other_endpoints_unchanged=57,
                  later_session_processes=9,all_qualified=raw['all_qualified'],
                  later_native_base_ms=raw['native_base_ms'],sessions_pooled=False,
                  copied_artifact_hashes=copied,source_data_sha256=sha(HERE/'data/full-trained-results.json'))
    (HERE/'data/verification.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()

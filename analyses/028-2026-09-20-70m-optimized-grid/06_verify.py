"""Audit preservation, arithmetic, complete coverage and installed artifact identity."""
import hashlib
import json
import math
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
RUN=ROOT/'runs/045-2026-09-20-pythia70m-kernel-grid'
DRAFT=ROOT/'manuscript/draft'
OLD=ROOT/'analyses/027-2026-09-20-run044-manuscript'


def read(path):return json.loads(path.read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    full=read(HERE/'data/full-trained-results.json')
    old=read(OLD/'data/full-trained-results.json')
    raw=read(RUN/'results/matched-grid.json')
    receipt=read(RUN/'results/local-verification.json')
    assert receipt['status']=='verified' and raw['status']=='complete'
    assert raw['fresh_processes']==78 and raw['checkpoints']==26
    assert raw['timings_per_implementation_checkpoint']==1344
    rows=full['trained_points']
    assert len(rows)==len({r['source_attempt'] for r in rows})==84
    assert {size:sum(r['model']==size for r in rows) for size in ('14M','70M','410M')}=={'14M':45,'70M':27,'410M':12}
    by_source={r['source_attempt']:r for r in rows}
    by_key={r['checkpoint_key']:r for r in rows if 'checkpoint_key' in r}
    for historical in old['trained_points']:
        current=by_source[historical['source_attempt']]
        mutable={'latency_ms','timing_session','kernel'} if current['model']=='70M' else set()
        assert all(current[k]==v for k,v in historical.items() if k not in mutable),historical['source_attempt']
    assert full['clipping_points']==old['clipping_points']
    for r in rows:
        assert math.isclose(r['sparsity'],100*r['zero_product_count']/r['model_product_count'],rel_tol=1e-12)
    by_id={r['run045_id']:r for r in rows if 'run045_id' in r}
    assert len(by_id)==26
    extra, = [r for r in rows if r.get('appendix_only')]
    assert extra['kappa']==.5 and extra['scope']=='hz' and extra['timing_session']=='Run046'
    assert extra['optimized_latency_ms'] is None and 'run045_id' not in extra
    figure=read(HERE/'data/70m-quality-sparsity-native-latency.json')
    assert len(figure['trained_points'])==26
    assert extra['checkpoint_key'] not in {r['checkpoint_key'] for r in figure['trained_points']}
    extra_table=(HERE/'tables/70m-additional-endpoint.tex').read_text()
    assert 'Optimized latency was not measured' in extra_table
    assert f"{extra['loss']:.4f} & {extra['sparsity']:.3f} & {extra['original_port_latency_ms']:.3f}" in extra_table
    for original in raw['conditions']:
        r=by_id[original['id']]
        assert r['implementation_latency_ms']==original['latency_ms']
        assert r['qualified']==original['qualified']
        assert r['loss']!=None and r['kernel']=='opt073'
        assert r['latency_ms']==(original['latency_ms']['candidate_graph'] if original['qualified']['candidate_graph'] else None)
        expected=raw['native_base_ms'] if original['id']=='c00' else r['latency_ms']
        assert r['displayed_latency_ms']==expected
    for pair in full['paired_pressure']:
        a=by_key[pair['treatment_key']];h=by_key[pair['reference_key']]
        assert pair['loss']==a['loss']-h['loss']
        assert pair['sparsity']==a['sparsity']-h['sparsity']
        expected=None if a['latency_ms'] is None or h['latency_ms'] is None else 1000*(a['latency_ms']-h['latency_ms'])
        assert pair['latency_us']==expected
    assert len(full['paired_pressure'])==20
    for path,digest in full['sources_sha256'].items():assert sha(ROOT/path)==digest,path
    names={'figures':['23-70m-quality-sparsity-native-latency.pdf','03-pressure-scope-threshold.pdf',
                     '20-kernel-structure-native-base-speedup.pdf','22-14m-main-quality-sparsity-latency.pdf'],
           'supplementary-data':['all-model-quality-sparsity.json','70m-quality-sparsity-native-latency.json',
                                 'paired-pressure-figure-data.json','kernel/70m-retained-controls.json']}
    copied={}
    for folder,files in names.items():
        sources=read(DRAFT/folder/'SOURCES.json')
        for name in files:
            record=sources[name]
            assert sha(DRAFT/folder/name)==sha(ROOT/record['source'])==record['sha256'],name
            copied[f'{folder}/{name}']=record['sha256']
    for name in ('14m-only.tex','14m-70m.tex','70m-kernel-transfer.tex','70m-additional-endpoint.tex'):
        assert (HERE/'tables'/name).read_bytes()==(DRAFT/'tables/compact-results'/name).read_bytes()
    # Check every displayed 70M endpoint against the verified reducer, including
    # both backends rather than just trusting the table's column headings.
    table=(HERE/'tables/14m-70m.tex').read_text()
    for row in by_id.values():
        fields=[f"{row['loss']:.4f}",f"{row['sparsity']:.3f}"]
        for mode in ('legacy_graph','candidate_graph'):
            fields.append(f"{row['implementation_latency_ms'][mode]:.3f}" if row['qualified'][mode] else r'$\dagger$')
        assert ' & '.join(fields) in table,row['run045_id']
    assert 'report all 84' in (DRAFT/'results-appendix.tex').read_text()
    assert 'Across 41 14M-parameter and 26 70M-parameter' in (DRAFT/'introduction.tex').read_text()
    best=min((r for r in by_id.values() if r['latency_ms'] is not None),key=lambda r:r['latency_ms'])
    facts=dict(checkpoints=84,matched_70m_checkpoints=26,fresh_processes=78,
               all_qualified=raw['all_qualified'],historical_endpoints_preserved=79,
               historical_clipping_unchanged=True,native_base_ms=raw['native_base_ms'],
               fastest_optimized={k:best[k] for k in ('run045_id','scope','pressure','kappa','loss','sparsity','latency_ms','native_base_speedup')},
               faster_local_pairs=full['matched_summary']['faster_local_pairs'],
               copied_artifact_hashes=copied,
               sources_checked=len(full['sources_sha256']),
               source_data_sha256=sha(HERE/'data/full-trained-results.json'))
    (HERE/'data/verification.json').write_text(json.dumps(facts,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(facts,indent=2))


if __name__=='__main__':main()

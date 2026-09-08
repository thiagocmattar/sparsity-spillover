"""Verify all 54 complete sweeps and expose full measurements plus simple tables."""
import csv
import gzip
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0,str(ROOT/'src'))
from sparsity_research.ceilings import architecture_ceiling
from clipping import nondominated


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build():
    inputs=json.loads((HERE/'input-manifest.json').read_text())
    sources={r['checkpoint_content_sha256']:r for r in inputs['checkpoints']}
    assert len(sources)==54
    evidence={}
    raw=[]

    def add(path,field):
        data=json.loads(path.read_text())
        evidence[path.relative_to(ROOT).as_posix()]=sha(path)
        for point in data[field]:
            source=sources[point['checkpoint_content_sha256']]
            raw.append({'source_checkpoint_id':source['id'],
                        'evidence_path':path.relative_to(ROOT).as_posix(),**point})

    add(next((ROOT/'analyses').glob('006-*/teal_all_variants.json')),'conditions')
    for run in ('018','019'):
        add(next((ROOT/'runs').glob(f'{run}-*/artifacts/teal/teal_frontiers.json')),'points')
    for source in inputs['checkpoints']:
        if source['evaluate']:
            path=HERE/'artifacts/attempts/001'/source['key']/'sweep.json'
            data=json.loads(path.read_text())
            assert data['status']=='complete_verified' and data['source']==source
            add(path,'points')
    assert len(raw)==540
    points=[]
    trained=[]
    families=('A0','A1-H','A1-H-L1','A1-H-OL1','A4','A4-OL1','A7','A7-OL1')
    ordered=sorted(inputs['checkpoints'],key=lambda r:(('14M','70M','410M').index(r['scale']),
                   families.index(r['family']),r['training_parameter'] or 0))
    for source in ordered:
        group=sorted([r for r in raw if r['source_checkpoint_id']==source['id']],
                     key=lambda r:r['target_sparsity'])
        assert [r['target_sparsity'] for r in group]==inputs['targets']
        baseline=group[0]['validation']['loss']
        assert abs(baseline-source['source_loss'])<=5e-4
        topology='A7-Z-POST' if source['family'].startswith('A7') else 'A4-Z'
        ceiling=architecture_ceiling(topology,**{k:source['ceiling'][k] for k in
                                     ('layers','hidden_size','ffn_size','sequence_length','vocabulary_size')})
        flags=nondominated(group)
        for rawpoint,flag in zip(group,flags):
            coverage=rawpoint['validation']
            assert all(coverage[k]==v for k,v in {'sequences':338,'input_tokens':692224,
                       'source_tokens':693668,'excluded_tail_tokens':1444,'complete_block_coverage':True}.items())
            counts=rawpoint['logical_products']
            assert sum(r['zero_product_count'] for r in counts['per_operation'].values())==counts['block_zero_product_count']
            assert sum(r['product_count'] for r in counts['per_operation'].values())==counts['block_product_count']
            assert counts['model_product_count']==338*ceiling['model_product_count']
            assert counts['R_model']==counts['block_zero_product_count']/counts['model_product_count']
            assert {k.split('.layer_')[0] for k in rawpoint['thresholds_by_site_layer']}==set(inputs['clipping_sites'])
            assert len(rawpoint['thresholds_by_site_layer'])==4*ceiling['layers']
            assert all(r['nonfinite']==0 for r in rawpoint['activation_rows'])
            p=rawpoint['target_sparsity']
            points.append({'id':source['id']+f':clip:{p}', 'kind':'clipped',
                           'source_checkpoint_id':source['id'],'scale':source['scale'],
                           'family':source['family'],'training_parameter':source['training_parameter'],
                           'dose':p,'loss':coverage['loss'],'delta_loss_from_p0':coverage['loss']-baseline,
                           'R_model':counts['R_model'],'counts':counts,'coverage':coverage,
                           'thresholds_by_site_layer':rawpoint['thresholds_by_site_layer'],
                           'ceiling':ceiling,'normalization_sites':ceiling['active_sites'],
                           'U_arch':counts['R_model']/ceiling['R_model_max_fraction'],
                           'nondominated_within_checkpoint':flag,'source':rawpoint['evidence_path'],
                           'checkpoint_content_sha256':source['checkpoint_content_sha256']})
        trained.append({'id':source['id'],'kind':'trained','scale':source['scale'],
                        'family':source['family'],'training_parameter':source['training_parameter'],
                        'loss':source['source_loss'],'R_model':source['source_R_model']})
    assert len({r['id'] for r in points})==540
    frontiers={}
    for scale in ('14M','70M','410M'):
        clipped=[r for r in points if r['scale']==scale]
        combined=[r for r in trained if r['scale']==scale]+clipped
        def ids(rows):
            proxy=[{'validation':{'loss':r['loss']},'logical_products':{'R_model':r['R_model']}} for r in rows]
            return [r['id'] for r,f in zip(rows,nondominated(proxy)) if f]
        frontiers[scale]={'clipping_only':ids(clipped),'training_and_clipping':ids(combined)}
    return {'schema_version':1,'status':'complete_verified','points':points,'trained':trained,
            'frontiers':frontiers,'sources':evidence,
            'protocol':{'targets':inputs['targets'],'clipping_sites':inputs['clipping_sites'],
                        'new_points':350,'reused_points':190,'checkpoints':54,
                        'normalization':'union of retained trained gate sites and evaluation clipping sites',
                        'interpretation':'One-seed evaluated quality-logical-sparsity frontiers; no interpolated models or measured speedups.'}},raw


def main():
    data,raw=build()
    out=HERE/'results'
    out.mkdir(exist_ok=True)
    (out/'clipping-points.json').write_text(json.dumps(data,indent=2)+'\n',newline='\n')
    with (out/'raw-points.json.gz').open('wb') as file:
        with gzip.GzipFile(fileobj=file,mode='wb',mtime=0,filename='') as stream:
            stream.write(json.dumps(raw,sort_keys=True,separators=(',',':')).encode())
    columns=['source_checkpoint_id','scale','family','training_parameter','clipping_target_p','loss',
             'delta_loss_from_p0','R_model','U_arch','nondominated_within_checkpoint']
    with (out/'clipping-points.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=columns,lineterminator='\n')
        writer.writeheader()
        writer.writerows({k:r['dose'] if k=='clipping_target_p' else r[k] for k in columns} for r in data['points'])
    (out/'frontiers.json').write_text(json.dumps(data['frontiers'],indent=2)+'\n',newline='\n')
    print(json.dumps({'points':len(data['points']),'frontier_counts':{
        k:len(v['training_and_clipping']) for k,v in data['frontiers'].items()}}))


if __name__=='__main__':
    main()

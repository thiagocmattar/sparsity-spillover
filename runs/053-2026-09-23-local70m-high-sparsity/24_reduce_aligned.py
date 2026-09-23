"""Apply the stated complete-path shortlist rule to retained paired samples."""
import argparse
import json
import math
from collections import defaultdict
from local_support import RUN,load,write,sha

def geometric(values):
    assert values and all(math.isfinite(x) and x>0 for x in values)
    return math.exp(math.fsum(math.log(x) for x in values)/len(values))

def main():
    p=argparse.ArgumentParser();p.add_argument('--tag',default='aligned-001');args=p.parse_args()
    cfg=load(RUN/'aligned-tile-config.json');cells={};sources=[];counts={};compiler={}
    operator=RUN/'artifacts/attempts'/(args.tag+'-operators')/'result.json'
    op=load(operator);assert op['status']=='passed'
    sources.append({'path':operator.relative_to(RUN).as_posix(),'sha256':sha(operator)})
    for cid in cfg['conditions']:
        folder=RUN/'artifacts/attempts'/(args.tag+'-'+cid)
        result=load(folder/'result.json')
        assert result['status']=='complete' and result['completed_blocks']==cfg['blocks']
        for name in ('result.json','samples.jsonl','numerics.json','work.json','compiler.json','activation-statistics.json','weights.json'):
            path=folder/name;sources.append({'path':path.relative_to(RUN).as_posix(),'sha256':sha(path)})
        numeric=defaultdict(list)
        for row in load(folder/'numerics.json'):
            split='development' if row['block']<cfg['development_blocks'] else 'confirmation'
            numeric[(split,row['site'],row['mode'])].append(row['pass'])
        samples=defaultdict(list)
        with (folder/'samples.jsonl').open(encoding='utf-8') as f:
            for line in f:
                row=json.loads(line);samples[(row['split'],row['site'],row['mode'])].append(row)
        counts[cid]=sum(map(len,samples.values()))
        assert counts[cid]==cfg['blocks']*12*12*cfg['passes']
        for key,rows in samples.items():
            assert len(rows)==16*cfg['passes'] and len(numeric[key])==16,key
            assert len({(r['block'],r['repeat']) for r in rows})==len(rows)
            cells[(cid,*key)]={'qualified':all(numeric[key]),'host_us':geometric([r['host_us'] for r in rows]),
                              'cuda_us':geometric([r['cuda_us'] for r in rows])}
        compiler[cid]=load(folder/'compiler.json')
    controls=('dense_native','dense_fused','dense_dot','prior')
    table=[];shortlist=[]
    for site in [f'{s}.{i}' for s in ('h','z') for i in range(6)]:
        for spec in cfg['candidates']:
            mode=spec['id']; comparisons=[]
            for cid in cfg['conditions']:
                for split in ('development','confirmation'):
                    valid_controls={name:cells[(cid,split,site,name)] for name in controls if cells[(cid,split,site,name)]['qualified']}
                    assert valid_controls, (cid,split,site)
                    best=min(valid_controls,key=lambda n:valid_controls[n]['host_us'])
                    value=cells[(cid,split,site,mode)];off=cells[(cid,split,site,mode+'_no_skip')]
                    ref=valid_controls[best]
                    comparisons.append({'condition':cid,'split':split,'best_control':best,
                        'candidate_qualified':value['qualified'],'no_skip_qualified':off['qualified'],
                        'host_ratio':value['host_us']/ref['host_us'],'cuda_ratio':value['cuda_us']/ref['cuda_us'],
                        'host_us':value['host_us'],'control_host_us':ref['host_us'],
                        'skip_saving_us':off['host_us']-value['host_us']})
            moderate=[r for r in comparisons if r['condition'] in ('c24','c25')]
            accepted=all(r['candidate_qualified'] and r['host_ratio']<=.95 for r in moderate)
            row={'site':site,'candidate':mode,'shortlisted':accepted,
                 'worst_moderate_ratio':max(r['host_ratio'] for r in moderate),'comparisons':comparisons}
            table.append(row)
            if accepted:shortlist.append({'site':site,'candidate':mode})
    report={'status':'complete','scope':cfg['interpretation'],'operator_checks':len(op['checks']),
            'graph_checks':op['changed_input_graph_checks'],'sample_counts':counts,'shortlist':shortlist,
            'comparisons':table,'sources':sources,
            'criterion':'Numerically qualified and >=5% faster than best qualified dense/prior control at both moderate kappas in both prefixes. Component shortlist only.',
            'compiler_spill_entries':{cid:{name:[k.get('spills') for k in kernels] for name,kernels in record.items()
                                         if any(k.get('spills') for k in kernels)} for cid,record in compiler.items()}}
    write(RUN/'results'/(args.tag+'-summary.json'),report)
    text=['# Wider sparse component screen','',report['criterion'],'',
          '| Site | Best candidate | Worst moderate ratio | Shortlisted |','|---|---|---:|---|']
    for site in dict.fromkeys(r['site'] for r in table):
        row=min((r for r in table if r['site']==site),key=lambda r:r['worst_moderate_ratio'])
        text.append(f'| {site} | {row["candidate"]} | {row["worst_moderate_ratio"]:.4f} | {row["shortlisted"]} |')
    text+=['','Ratios below1 are faster. These are complete component paths on local training inputs, not full-model or RTX5090 results.','Sources: `24_reduce_aligned.py` and the source-hashed summary JSON.']
    (RUN/'results'/(args.tag+'-table.md')).write_text('\n'.join(text)+'\n',encoding='utf-8')
    print('\n'.join(text))

if __name__=='__main__':main()

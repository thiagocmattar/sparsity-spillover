"""Re-express retained Run042 controls using the paper's geometric-mean estimand."""
import hashlib
import json
import math
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
RUN=ROOT/'runs/042-2026-09-20-pythia70m-sparse-scale'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def gm(values):return math.exp(math.fsum(map(math.log,values))/len(values))


def main():
    summary=json.loads((RUN/'results/summary.json').read_text())
    selected={'opt073','selected-native-hz','selected-native-attention','selected-native-norm',
              'selected-native-rope','selected-native-head','selected-no-skip'}
    groups={};sources={};devices=set()
    for item in summary['sources']:
        path=RUN/item['path']
        assert sha(path)==item['sha256']
        if path.name!='result.json':continue
        result=json.loads(path.read_text())
        if result['validation_blocks']!=338 or result['condition'] not in ('c00','c21') or result['candidate'] not in selected:continue
        assert result['qualified'] and result['status']=='complete'
        if result['condition']=='c00' and result['candidate'] not in ('opt073','selected-native-hz'):continue
        timing=path.parent/'timing.json';data=json.loads(timing.read_text())
        group=groups.setdefault((result['condition'],result['candidate']),{'processes':0,'native':[],'candidate':[]})
        group['processes']+=1
        assert len(data['indices'])==64
        for mode,field in [('native_graph','native'),('candidate_graph','candidate')]:
            rows=[r for r in data['samples'] if r['mode']==mode]
            assert len(rows)==448 and all(r['output_shape']==[1,2048,50304] for r in rows)
            group[field].extend(r['host_ms'] for r in rows)
        devices.add(result['runtime']['device_uuid'])
        sources[path.relative_to(ROOT).as_posix()]=sha(path)
        sources[timing.relative_to(ROOT).as_posix()]=sha(timing)
    assert len(groups)==9 and len(devices)==1 and all(g['processes']==3 for g in groups.values())
    rows=[dict(condition=c,implementation=m,native_ms=gm(g['native']),candidate_ms=gm(g['candidate']),
               samples_per_implementation=len(g['candidate'])) for (c,m),g in groups.items()]
    base=next(r for r in rows if (r['condition'],r['implementation'])==('c00','opt073'))
    sparse=next(r for r in rows if (r['condition'],r['implementation'])==('c21','opt073'))
    dense=next(r for r in rows if (r['condition'],r['implementation'])==('c00','selected-native-hz'))
    result=dict(rows=rows,sources_sha256=sources,device_uuid=next(iter(devices)),
                native_base_ms=base['native_ms'],sparse_ms=sparse['candidate_ms'],alternate_dense_ms=dense['candidate_ms'],
                sparse_vs_alternate_dense_speedup=dense['candidate_ms']/sparse['candidate_ms'],
                estimand='Geometric mean of 1344 raw host observations per implementation/condition; separate Run042 session.')
    (HERE/'data/retained-controls.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('native_base_ms','sparse_ms','alternate_dense_ms','sparse_vs_alternate_dense_speedup')}))


if __name__=='__main__':main()

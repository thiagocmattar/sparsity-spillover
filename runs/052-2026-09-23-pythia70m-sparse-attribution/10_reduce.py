"""Report complete qualified comparisons; preserve failures and signed effects."""
import argparse,math
import numpy as np
from support import RUN,read,write,sha
from effects import interval,paired_intervals


def main():
    p=argparse.ArgumentParser();p.add_argument('--attempt',required=True);p.add_argument('--sparse-mode',choices=('candidate_graph','prior_c_graph'),default='candidate_graph');a=p.parse_args()
    records={};arrays={};source=[];indices=None;identities={};source_identity=None;data_identity=None
    for cid in ('c00','c24','c25','d05','d10'):
        records[cid]=[]
        for rep in (1,2,3):
            root=RUN/'artifacts'/f'{a.attempt}-{cid}-r{rep}'
            r=read(root/'result.json');q=read(root/'quality.json');t=read(root/'timing.json')
            assert r['status']=='complete' and q['blocks']==338 and q['prediction_tokens']==691886
            assert r['arguments']['condition']==cid and r['arguments']['replicate']==rep
            assert r['arguments']['phase'] in ('reference','final')
            identity=(r['checkpoint'],r['selection'],r['policies'])
            if cid not in identities:identities[cid]=identity
            assert identities[cid]==identity,'Checkpoint/policy changed between process replicates'
            if source_identity is None:source_identity=r['source_hashes'];data_identity=r['data_identity']
            assert source_identity==r['source_hashes'] and data_identity==r['data_identity'],'Unmatched source or validation data'
            if indices is None:indices=t['indices']
            assert t['indices']==indices and len(indices)==64
            for name in ('result.json','quality.json','timing.json'):source.append({'path':(root/name).relative_to(RUN).as_posix(),'sha256':sha(root/name)})
            modes={s['mode'] for s in t['samples']}
            cell={'loss':q['loss'],'qualified':q['pass'],'latency_ms':{}}
            for mode in sorted(modes):
                buckets={i:[] for i in range(64)};seen=set()
                for sample in t['samples']:
                    if sample['mode']!=mode:continue
                    key=(sample['repeat'],sample['input_index']);assert key not in seen;seen.add(key)
                    buckets[sample['input_index']].append(sample['host_ms'])
                assert all(len(v)==7 for v in buckets.values())
                vector=np.array([math.exp(math.fsum(map(math.log,v))/7) for v in buckets.values()])
                arrays.setdefault((cid,mode),[]).append(vector)
                cell['latency_ms'][mode]=float(np.exp(np.log(vector).mean()))
            records[cid].append(cell)
    chosen={'14_sparse05':('d05','k050_graph'),'14_sparse10':('d10','k050_graph'),
            '14_dense05':('d05','dense_policy_graph'),'14_dense10':('d10','dense_policy_graph'),
            '70_sparse05':('c24',a.sparse_mode),'70_sparse10':('c25',a.sparse_mode),
            '70_dense05':('c24','dense_policy_graph'),'70_dense10':('c25','dense_policy_graph')}
    qualification={key:all(r['qualified'][mode] for r in records[cid]) for key,(cid,mode) in chosen.items()}
    qualified=all(qualification.values())
    summary={'attempt':a.attempt,'records':records,'sources':source,'contrast_qualified':qualified,
             'dense_adjustment':'Predeclared dense_policy at each size, not a validation-selected kernel.',
             'effects':interval({k:arrays[v] for k,v in chosen.items()},qualified=qualification)}
    base=float(np.exp(np.mean(np.log(arrays[('c00','native_graph')]))))
    summary['native_Base70_ms']=base
    summary['same_checkpoint_controls']={}
    for cid in ('c24','c25'):
        m={mode:float(np.exp(np.mean(np.log(value)))) for (c,mode),value in arrays.items() if c==cid}
        sparse=m[a.sparse_mode]
        local={mode:arrays[(cid,mode)] for mode in m};local['Base70']=arrays[('c00','native_graph')]
        accepted={mode:all(r['qualified'][mode] for r in records[cid]) for mode in m}
        accepted['Base70']=all(r['qualified']['native_graph'] for r in records['c00'])
        differences={'saving_over_Base_ms':{'Base70':1,a.sparse_mode:-1}}
        for mode in ('native_graph','native_hz_graph','dense_fused_graph','dense_policy_graph'):
            differences['gain_over_'+mode+'_ms']={mode:1,a.sparse_mode:-1}
        row={'latency_ms':m}
        if a.sparse_mode=='candidate_graph':
            differences.update({
                'conditional_h_saved_ms':{'candidate_h_off_graph':1,a.sparse_mode:-1},
                'conditional_z_saved_ms':{'candidate_z_off_graph':1,a.sparse_mode:-1},
                'joint_saved_ms':{'candidate_hz_off_graph':1,a.sparse_mode:-1},
                'interaction_ms':{'candidate_hz_off_graph':1,'candidate_h_off_graph':-1,'candidate_z_off_graph':-1,a.sparse_mode:1}})
        row['effects']=paired_intervals(local,differences,ratios={'versus_native_Base_ratio':('Base70',a.sparse_mode)},qualified=accepted)
        summary['same_checkpoint_controls'][cid]=row
    summary['interpretation']='Qualification is required per contrast; failed contrasts retain descriptive points but have no inference interval. Dense/skipping gains and cross-kappa response are separate. Intervals are marginal 95%, not simultaneous family-wise coverage. No manuscript claims are emitted automatically.'
    write(RUN/'results'/f'{a.attempt}-summary.json',summary)
    print({'contrast_qualified':qualified,'effects':summary['effects'],'native_Base70_ms':base})


if __name__=='__main__':main()

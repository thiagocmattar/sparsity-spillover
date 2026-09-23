"""Training-only component promotion against the existing complete paths."""
import argparse,math
from support import RUN,read,write,sha


def select(screen,prior,margin=.05):
    cells=[f'{cid}:{split}:' for cid in ('c24','c25') for split in ('development','confirmation')]
    dense={};candidate={};details={}
    for key in prior:
        rows=[screen[c+key] for c in cells]
        names=set.intersection(*(set(r) for r in rows))
        valid={n for n in names if all(r[n]['qualified'] for r in rows)}
        dense_names=valid & {'dense_native','dense_fused','dense_dot'}
        if not dense_names:raise ValueError(f'No qualified dense control: {key}')
        d=min(dense_names,key=lambda n:(max(r[n]['host_ms']/r['dense_native']['host_ms'] for r in rows),n))
        dense[key]=d
        baseline=prior[key] if prior[key].startswith('c_') else d
        if baseline not in valid:raise ValueError(f'Prior route no longer qualifies: {key} {baseline}')
        scores={n:max(r[n]['host_ms']/r[baseline]['host_ms'] for r in rows)
                for n in valid if n.startswith(('e_','f_'))}
        best=min(scores,key=lambda n:(scores[n],n)) if scores else None
        chosen=best if best and scores[best]<=1-margin else baseline
        candidate[key]=chosen
        details[key]={'dense':d,'baseline':baseline,'scores':scores,'choice':chosen}
    ablations={}
    for disabled in ('h','z','hz'):
        ablations['candidate_'+disabled+'_off']={
            k:v+'_noskip' if k[0] in disabled and v.startswith(('c_','e_','f_')) else v
            for k,v in candidate.items()}
    return {'dense':dense,'policies':{'prior_c':prior,'candidate':candidate},
            'ablations':ablations,'details':details,
            'changed_sites':[k for k,v in candidate.items() if v!=details[k]['baseline']],
            'selection_data':'training blocks0:64 and64:128 at both kappas; no validation selection',
            'interpretation':'Component screen only; full-model training qualification and both-endpoint improvement are required before final freeze.'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--screen',required=True);p.add_argument('--output',default='selection-training.json');a=p.parse_args()
    path=RUN/'artifacts'/a.screen/'summary.json'
    result=select(read(path),read(RUN/'provenance/prior-selection.json')['policies']['sparse_c'],read(RUN/'config.json')['component_minimum_improvement'])
    result['screen_source']={'path':path.relative_to(RUN).as_posix(),'sha256':sha(path)}
    target=RUN/'provenance'/a.output
    if target.exists():raise FileExistsError(target)
    write(target,result);print({'changed_sites':result['changed_sites'],'policies':result['policies']})


if __name__=='__main__':main()

"""Seal a qualified K050 study; immutable provenance, not authorization."""
import argparse
from datetime import datetime,timezone
from common import RUN,ROOT,record,read_json,verify_record,write_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--verify',action='store_true');args=p.parse_args()
    path=RUN/'final-policy-002.json'
    if args.verify:
        policy=read_json(path)
        for row in policy['sources']+policy['dependencies']['files']:verify_record(row)
        print({'verified_sources':len(policy['sources']),'policy':record(path)});return
    if path.exists():raise FileExistsError(path)
    original=read_json(RUN/'final-policy-001.json')
    for row in original['sources']+original['dependencies']['files']:verify_record(row)
    qualifications=[]
    for condition in read_json(RUN/'config.json')['development_conditions']:
        folder=RUN/'artifacts'/f'k050-{condition}-graph-001'
        result=read_json(folder/'result.json');quality=read_json(folder/'quality.json')
        if result['status']!='complete' or quality['blocks']!=338:raise ValueError('Incomplete development qualification')
        for mode in ['native_graph','selected_graph','no_skip_graph','attention_dense_graph']:
            if not result['qualified'][mode] or not quality['pass'][mode]:raise ValueError('Unqualified selected/control mode')
        for row in read_json(folder/'manifest.json')['sources']:verify_record(row)
        qualifications += [record(folder/name) for name in ['result.json','quality.json','manifest.json']]
    primitive=RUN/'artifacts/k050-norm-001/result.json';result=read_json(primitive)
    if result['status']!='complete' or result['cases']!=28 or result['graph_refresh_cases']!=3:raise ValueError('Primitive qualification incomplete')
    files=[ROOT/r['path'] for r in original['sources']]
    files += [RUN/name for name in ['115_hybrid_diagnostics.py','116_hybrid_study.py','117_freeze_hybrid_study.py']]
    for candidate in ['k042','k049','k050']:
        files += [f for f in (RUN/f'candidates/{candidate}').iterdir() if f.is_file()]
    write_json(path,{'created_utc':datetime.now(timezone.utc).isoformat(),'candidate':'k050','shortcut':False,
        'selection':'Uniform K050 following ten registered development endpoints; no checkpoint-dependent fallback',
        'scope':'Six matmul families with fragment-skipping attention/a/m and hybrid SIMT/padded-MMA h/z; paired normalization fusion',
        'limits':'Not arbitrary scalar-zero elimination; h/z dense control has twofold row padding; bypassed MMA includes SIMT substitution; attention skipping may remain slower',
        'study':'All35 checkpoints;64 fixed validation timing identities x7 passes x3 fresh processes; full338 validation and once-per-checkpoint diagnostics; nine modes including unfused K049 graph',
        'qualification':[record(primitive)]+qualifications,'previous_policy':record(RUN/'final-policy-001.json'),
        'sources':[record(f) for f in sorted(set(files))],'dependencies':original['dependencies']})
    print(record(path))


if __name__=='__main__':main()

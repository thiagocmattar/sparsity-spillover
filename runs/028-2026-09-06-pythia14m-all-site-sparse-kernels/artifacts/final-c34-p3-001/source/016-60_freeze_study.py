"""Seal this study's source identity; this is provenance, not authorization."""
import argparse
from datetime import datetime,timezone
from common import RUN,ROOT,R25,R26,R27,record,read_json,verify_record,write_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--verify',action='store_true')
    p.add_argument('--output',default='final-policy-001.json');a=p.parse_args()
    if '/' in a.output or '\\' in a.output or not a.output.endswith('.json'):p.error('Local policy basename required')
    path=RUN/a.output
    if a.verify:
        policy=read_json(path)
        for row in policy['sources']:verify_record(row)
        for row in policy['dependencies']['files']:verify_record(row)
        print({'verified_sources':len(policy['sources']),'policy':record(path)});return
    if path.exists():raise FileExistsError(path)
    files=[RUN/name for name in ['config.json','common.py','23_dependencies.py','45_graph_forward.py','58_study_diagnostics.py','59_study.py','60_freeze_study.py']]
    for candidate in ['k020','k021','k031','k032','k033','k035','k036']:
        files += [f for f in (RUN/f'candidates/{candidate}').iterdir() if f.is_file()]
    files += [R27/name for name in ['adapter.py','kernel.cu','run027_common.py']]
    files += [R25/name for name in ['run025_common.py','measurement.py','config.json','autoresearch/dense_probe/probe.py']]
    files += [R26/f'autoresearch/candidates/{candidate}/{name}' for candidate in ['k018','k019'] for name in ['candidate.py','kernel.cu']]
    files += list((ROOT/'src').rglob('*.py'))
    write_json(path,{'created_utc':datetime.now(timezone.utc).isoformat(),'candidate':'k036','shortcut':False,
        'selection':'uniform no-prefix policy after paired development controls; no checkpoint-dependent fallback',
        'scope':'all six matmul families;16x8x16 zero-A or attention zero-operand MMA skipping, not every scalar zero',
        'limits':'attention skipping was individually slower in development; previous graph path was faster at high R',
        'study':'all35 checkpoints,64 validation timing identities x7 passes x3 fresh processes; full338 validation and once-per-checkpoint diagnostics',
        'sources':[record(f) for f in sorted(set(files))],
        'dependencies':read_json(RUN/'runtime/vendor/inventory.json')})
    print(record(path))


if __name__=='__main__':main()

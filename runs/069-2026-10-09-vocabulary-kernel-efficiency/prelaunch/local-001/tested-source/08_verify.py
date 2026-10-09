"""Verify final coverage, bounds, process completeness and retrievable file identities."""
import argparse
from common import RUN, REPO, config, sources, read, record, verify, write


def main():
    p=argparse.ArgumentParser();p.add_argument('--transfer-only',action='store_true');args=p.parse_args()
    cfg=config();results=[]
    for source in sources()['checkpoints']:
        for replicate in range(1,cfg['process_replicates']+1):
            folder=RUN/'artifacts/endpoints'/f'{source["id"]}-r{replicate}'
            result=read(folder/'result.json');quality=read(folder/'quality.json')
            if result['status']!='complete' or not result['qualified'] or result['evidence']!='valid':
                raise ValueError('Incomplete or development endpoint')
            if result['blocks']!=338 or result['predictions']!=691886 or quality['excluded_tail_tokens']!=1444:
                raise ValueError('Incomplete validation')
            if quality['bounds']!=cfg['numerical_bounds']:raise ValueError('Changed acceptance bounds')
            for mode in result['timings']['full_forward']:
                checks=quality['checks'][mode]
                if len(checks)!=338 or not all(c['pass'] for c in checks):raise ValueError('Unqualified mode')
                if abs(quality['loss'][mode]-quality['loss']['anchor'])>cfg['numerical_bounds']['validation_loss_atol']:
                    raise ValueError('Pooled loss mismatch')
                for boundary in result['timings']:
                    if result['timings'][boundary][mode]['count']!=cfg['timing_inputs']*cfg['timing_passes']:
                        raise ValueError('Incomplete timing samples')
            results.append(record(folder/'result.json'))
    for row in read(RUN/'provenance/upload-inventory.json'):verify(row)
    write(RUN/'results/verification.json',{'status':'verified','processes':len(results),'results':results,
          'limitations':'Does not establish a practical quality-latency frontier gain or restore trained quality.'})
    paths=[p for directory in ('artifacts','results','prelaunch') for p in (RUN/directory).rglob('*')
           if p.is_file() and p.suffix not in ('.tmp','.pid') and p.name not in ('transfer-inventory.json',)]
    rows=[record(p) for p in sorted(paths)]
    write(RUN/'transfer/transfer-inventory.json',rows)
    print({'status':'verified','processes':len(results),'transfer_files':len(rows)},flush=True)


if __name__=='__main__':main()

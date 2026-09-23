"""Summarize the retained local smoke without promoting it to final evidence."""
import argparse
from local_support import RUN, load, sha, write

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--tag', default='smoke-002')
    args=parser.parse_args()
    cfg=load(RUN/'config.json')
    rows=[]; profiles={}; sources=[]
    for cid in cfg['conditions']:
        folder=RUN/'artifacts/attempts'/(args.tag+'-'+cid)
        for name in ('result.json','quality.json','memory.json','timing.json','profile-summary.json'):
            path=folder/name
            sources.append({'path':path.relative_to(RUN).as_posix(),'sha256':sha(path)})
        result=load(folder/'result.json'); quality=load(folder/'quality.json')
        assert result['status']=='complete' and all(quality['pass'].values())
        assert quality['split']=='development' and quality['blocks']==cfg['smoke_blocks']
        timing=load(folder/'timing.json')
        assert len(timing['samples'])==len(cfg['modes'])*cfg['smoke_blocks']*cfg['smoke_passes']
        row={'condition':cid,'kappa':result['checkpoint']['dose'],
             'training_smoke_loss':result['loss']['native'],
             'timing_ms':result['timing_ms'],'peak_allocated_GiB':result['peak_allocated_bytes']/1024**3,
             'peak_reserved_GiB':result['peak_reserved_bytes']/1024**3,
             'minimum_free_after_capture_GiB':min(r['free_bytes'] for r in result['memory'])/1024**3,
             'numerical_blocks_per_second':cfg['smoke_blocks']/result['numerical_seconds'],
             'elapsed_seconds':result['elapsed_seconds']}
        rows.append(row)
        profiles[cid]=[]
        for profile in load(folder/'profile-summary.json')['profiles']:
            if profile['execution']!='graph':continue
            kernels=sorted(profile['kernels'].items(),key=lambda pair:pair[1]['duration_us'],reverse=True)
            profiles[cid].append({'mode':profile['mode'],'inputs':profile['inputs'],
                'total_instrumented_kernel_us_per_input':sum(v['duration_us'] for _,v in kernels)/profile['inputs'],
                'top_kernels':[{'name':name,'calls':v['calls'],'us_per_input':v['duration_us']/profile['inputs']} for name,v in kernels[:8]]})
    report={'scope':'Four training blocks, two passes, one process per checkpoint; calibration only.',
            'tag':args.tag,'conditions':rows,'profiles':profiles,'sources':sources,
            'full_reference_numerical_seconds_extrapolated':sum(338/r['numerical_blocks_per_second'] for r in rows),
            'limitations':['No full-validation qualification in this smoke.',
                           'No statistical inference or cross-size comparison.',
                           'Profiles are instrumented; their durations are not benchmark latency.',
                           'Local laptop timings are separate from RTX5090 manuscript evidence.']}
    write(RUN/'results'/f'{args.tag}-summary.json',report)
    table=['# Local 70M smoke (calibration only)','',report['scope'],'',
           '| Checkpoint | Native | Original port | opt073 | opt073 h/z skip off | Native h/z replacement | Peak allocated GiB | Min. free after capture GiB |',
           '|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        times=row['timing_ms']
        table.append('| '+row['condition']+' | '+' | '.join(f'{times[m+"_graph"]:.4f}' for m in cfg['modes'])+f' | {row["peak_allocated_GiB"]:.2f} | {row["minimum_free_after_capture_GiB"]:.2f} |')
    table += ['', 'Latencies are geometric-mean synchronized host milliseconds. c00=Base; c24/c25/c26=T2/Ph kappa .05/.1/.5.',
              'All cells passed only the four-block numerical smoke. These are not final speedup or quality claims.',
              'Source: `05_reduce.py`, with per-file hashes in the accompanying summary JSON.']
    (RUN/'results'/f'{args.tag}-table.md').write_text('\n'.join(table)+'\n',encoding='utf-8')
    print('\n'.join(table))

if __name__=='__main__':main()

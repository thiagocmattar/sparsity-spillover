"""Verify final coverage, bounds, process completeness and retrievable file identities."""
from common import RUN, REPO, config, sources, read, record, verify, write


def main():
    cfg=config();results=[]
    gpu_uuids=set()
    commits=set()
    def identity(value):
        if value['development']:raise ValueError('Development source identity')
        verify(value['inventory']);commits.add(value['commit'])
    def gpu(value):
        if value['gpu']!='NVIDIA GeForce RTX 5090':raise ValueError('Wrong final GPU')
        gpu_uuids.add(value['gpu_uuid'])
    for source in sources()['checkpoints']:
        for replicate in range(1,cfg['process_replicates']+1):
            folder=RUN/'artifacts/endpoints'/f'{source["id"]}-r{replicate}'
            result=read(folder/'result.json');quality=read(folder/'quality.json')
            gpu(result['runtime'])
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
            identity(result['identity']);verify(result['snapshot'])
            results.append(record(folder/'result.json'))
    for row in read(RUN/'provenance/upload-inventory.json'):verify(row)
    capture=read(RUN/'artifacts/capture/manifest.json')
    identity(capture['identity']);gpu(capture['runtime'])
    if capture['status']!='complete' or len(capture['indices'])!=cfg['timing_inputs']:
        raise ValueError('Incomplete operand capture')
    for row in capture['files']:verify(row)
    for source in sources()['checkpoints']:
        screen=read(RUN/'artifacts/screen'/source['id']/'result.json')
        if screen['status']!='complete':raise ValueError('Incomplete projection screen')
        identity(screen['identity']);gpu(screen['runtime'])
        for mode,counts in screen['work_counts'].items():
            if counts['input_count']!=cfg['timing_inputs'] or counts['issued_mma']+counts['skipped_mma']!=counts['potential_mma']:
                raise ValueError('Incomplete real-operand MMA counts')
        verify(screen['input_snapshot'])
    profile=read(RUN/'artifacts/profiles/manifest.json')
    for report in profile['reports']:
        if report.get('runtime_identity'):
            identity(report['runtime_identity']['identity']);gpu(report['runtime_identity']['runtime'])
    if len(gpu_uuids)!=1 or 'unavailable' in gpu_uuids:raise ValueError('Same physical GPU identity required')
    if len(commits)!=1 or not next(iter(commits)):raise ValueError('Consistent committed source identity required')
    write(RUN/'results/verification.json',{'status':'verified','processes':len(results),'results':results,
          'profiler_status':profile['status'],
          'limitations':'Does not establish a practical quality-latency frontier gain or restore trained quality. Unavailable counters remain missing evidence.'})
    paths=[p for directory in ('artifacts','results','prelaunch') for p in (RUN/directory).rglob('*')
           if p.is_file() and p.suffix not in ('.tmp','.pid') and p.name not in ('transfer-inventory.json',)]
    rows=[record(p) for p in sorted(paths)]
    write(RUN/'transfer/transfer-inventory.json',rows)
    print({'status':'verified','processes':len(results),'transfer_files':len(rows)},flush=True)


if __name__=='__main__':main()

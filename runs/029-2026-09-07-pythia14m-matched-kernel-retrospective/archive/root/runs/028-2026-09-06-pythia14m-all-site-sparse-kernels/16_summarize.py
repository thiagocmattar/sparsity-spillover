"""Read-only compact summary of terminal development evidence."""
import json
from statistics import median
from common import RUN,read_json

for folder in sorted((RUN/'artifacts').iterdir()):
    result_path=folder/'result.json'
    if not result_path.exists():continue
    r=read_json(result_path)
    if r.get('timing'):
        print(json.dumps({'attempt':folder.name,'status':r['status'],'qualified':r['qualified'],
            'speedup':{k:v['paired_geomean_speedup'] for k,v in r['timing'].items()},
            'host_ms':{k:v['median_host_ms'] for k,v in r['timing'].items()},
            'loss':r['loss'],'loss_delta':r['loss_delta'],'validation_blocks':r['validation_blocks']}))
    if '-real-' in folder.name and (folder/'components.json').exists():
        timings=read_json(folder/'timings.json')
        for row in read_json(folder/'components.json'):
            if row['identity'].startswith('synthetic'):continue
            times={mode:median(x['host_ms'] for x in timings if x['identity']==row['identity'] and x['mode']==mode)
                   for mode in {x['mode'] for x in timings}}
            print(json.dumps({'attempt':folder.name,'identity':row['identity'],
                'zero_query_fraction':row['zero_query_rows']/row['query_rows'],
                'qkv_zero_fractions':{s:row[s+'_zero_elements']/row['elements'] for s in ['q','k','v']},
                'times_ms':times,'prefix_round0':row['variants']['prefix1-round0']}))

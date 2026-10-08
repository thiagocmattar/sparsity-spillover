"""Strict cohort verification and additive event/CUPTI budgets."""
import collections
import math
import statistics
from accounting import budget,intervals_union
from trace_copy_mapping import map_native_copies
from runio import ROOT,read,write,record

def lines(path):return [__import__('json').loads(s) for s in path.read_text().splitlines()]
def describe(values):
    values=sorted(values);n=len(values)
    return {'count':n,'mean':statistics.mean(values),'geomean':math.exp(statistics.mean(map(math.log,values))) if min(values)>0 else None,'median':statistics.median(values),'p10':values[int((n-1)*.1)],'p90':values[int((n-1)*.9)],'min':min(values),'max':max(values)}

def main():
    cfg=read(ROOT/'config.json');out={'status':'verified','conditions':{},'sources':[]};uuids=set()
    for size in cfg['sizes']:
        for pair in cfg['pairs']:
            processes=[];base=collections.defaultdict(list);staged=collections.defaultdict(list);traces=[]
            for replicate in range(1,4):
                dest=ROOT/'artifacts/attempts'/f'full-{size}-k{pair}-r{replicate}-001'
                result=read(dest/'result.json');q=read(dest/'quality.json')
                if result['status']!='complete' or not result['qualified'] or not all(q['pass'].values()):raise ValueError('Unqualified '+str(dest))
                if (q['blocks'],q['documents'],q['prediction_tokens'],q['excluded_tail_tokens'])!=(338,500,691886,1444):raise ValueError('Coverage')
                comparisons=[r for rows in q['gates'].values() for r in rows]
                if len(q['gates'])!=6 or len(comparisons)!=338*6:raise ValueError('Incomplete numerical comparisons')
                a=lines(dest/'baseline-samples.jsonl');b=lines(dest/'stage-samples.jsonl')
                if len(a)!=64*7*3 or len(b)!=len(a):raise ValueError('Timing coverage')
                uuids.add(result['runtime']['gpu_uuid'])
                processes.append({'replicate':replicate,'gpu_uuid':result['runtime']['gpu_uuid'],'loss':result['loss'],'logit_comparisons':len(comparisons),'max_logit_absolute_error':max(r['max_abs'] for r in comparisons),'max_relative_l2':max(r['relative_l2'] for r in comparisons),'peak_allocated_bytes':result['peak_allocated_bytes'],'elapsed_seconds':result['elapsed_seconds']})
                processes[-1]['timing']={}
                for mode in cfg['modes']:
                    ar=[r for r in a if r['mode']==mode];br=[r for r in b if r['mode']==mode]
                    if len(ar)!=448 or len(br)!=448:raise ValueError('Mode timing coverage')
                    if {r['input_index'] for r in ar}!=set(range(64)):raise ValueError('Input coverage')
                    expected={(i,r) for i in range(64) for r in range(7)}
                    if {(r['input_index'],r['repeat']) for r in ar}!=expected or {(r['input_index'],r['repeat']) for r in br}!=expected:raise ValueError('Paired pass coverage')
                    if any(r['output_shape']!=[1,2048,50304] for r in ar+br):raise ValueError('Output shape')
                    processes[-1]['timing'][mode]={'host_us':describe([r['host_ms']*1000 for r in ar]),'cuda_us':describe([r['cuda_ms']*1000 for r in ar])}
                    base[mode].extend({**r,'replicate':replicate} for r in ar);staged[mode].extend(br)
                if replicate==1:
                    traces=read(dest/'traces.json');diagnostics=read(dest/'diagnostics.json')
                    if len(traces)!=3*4 or diagnostics['status']!='complete' or diagnostics['coverage']['blocks']!=338:raise ValueError('Diagnostic/trace coverage')
                for p in sorted(dest.glob('*.json'))+sorted(dest.glob('*.jsonl')):out['sources'].append(record(p))
            modes={}
            for mode in cfg['modes']:
                names=staged[mode][0]['stage_names']
                if any(r['stage_names']!=names for r in staged[mode]):raise ValueError('Stage signature changed')
                events=budget(names,[r['stage_us'] for r in staged[mode]])
                stage_samples=collections.defaultdict(list);nodes={};replays=[];mappings=[]
                for trace in [t for t in traces if t['mode']==mode]:
                    m=map_native_copies(trace)
                    if not m['mapped']:raise ValueError(f'Unmapped {size}/{pair}/{mode}: {m}')
                    count=m['kernels_per_replay'];rows=m['kernels']
                    if trace['graph']['repeats']!=3 or len(rows)!=3*count:raise ValueError('Trace replay coverage')
                    mappings.append({'input_index':trace['input_index'],'method':m['mapping'],'copy_nodes':m.get('graph_only_gate_copies',[])})
                    for replay in range(3):
                        subset=rows[replay*count:(replay+1)*count];by_stage=collections.defaultdict(float)
                        for i,r in enumerate(subset):
                            if r['stage'] not in names:raise ValueError('Unknown mapped stage')
                            by_stage[r['stage']]+=r['duration_us'];node=nodes.setdefault(i,{'index':i,'stage':r['stage'],'kernel':r['name'],'samples':[]})
                            if (node['stage'],node['kernel'])!=(r['stage'],r['name']):raise ValueError('Kernel signature changed across inputs')
                            node['samples'].append(r['duration_us'])
                        for name in names:stage_samples[name].append(by_stage[name])
                        active=intervals_union([(r['ts_us'],r['ts_us']+r['duration_us']) for r in subset]);span=max(r['ts_us']+r['duration_us'] for r in subset)-min(r['ts_us'] for r in subset)
                        replays.append({'active_us':active,'span_us':span,'gap_us':span-active,'kernel_count':count,'summed_kernel_us':sum(r['duration_us'] for r in subset)})
                active=statistics.mean(r['active_us'] for r in replays)
                stages=[{'stage':name,**describe(values),'fraction_of_active':statistics.mean(values)/active} for name,values in stage_samples.items()]
                if abs(sum(r['mean'] for r in stages)-active)>.1:raise ValueError('Overlapping stages require separate accounting')
                tracebudget={'all_signatures_match':True,'mappings':mappings,'replays':replays,'stages':stages,'mean_active_us':active,'mean_span_us':statistics.mean(r['span_us'] for r in replays),'mean_gap_us':statistics.mean(r['gap_us'] for r in replays),'kernel_nodes':[{k:v for k,v in n.items() if k!='samples'}|describe(n['samples']) for n in nodes.values()]}
                host=describe([r['host_ms']*1000 for r in base[mode]]);cuda=describe([r['cuda_ms']*1000 for r in base[mode]])
                ihost=describe([r['host_ms']*1000 for r in staged[mode]]);icuda=describe([r['cuda_ms']*1000 for r in staged[mode]])
                reference=[r['cuda_ms']*1000 for r in base[mode] if r['replicate']==1 and r['input_index']<4]
                tracebudget.update(matched_input_unprofiled_outer_cuda_us=describe(reference),span_over_matched_outer_cuda_ratio=tracebudget['mean_span_us']/statistics.mean(reference))
                modes[mode]={'baseline_host_us':host,'baseline_cuda_us':cuda,'equal_staging_us':describe([r['staging_ms_excluded']*1000 for r in base[mode]]),'host_minus_cuda_mean_us':host['mean']-cuda['mean'],'instrumented_host_us':ihost,'instrumented_cuda_us':icuda,'event_instrumentation_cuda_mean_ratio':icuda['mean']/cuda['mean'],'event_partition':events,'trace_budget':tracebudget}
            key=f'{size}-k{pair}';native=modes['native']['baseline_host_us']['geomean'];sparse=modes['sparse']['baseline_host_us']['geomean'];noskip=modes['no_skip']['baseline_host_us']['geomean']
            counter_path=ROOT/'artifacts'/f'{"feature-unions-70m" if size=="70m" else "short-rows-31m"}-k{pair}.json';counters=read(counter_path)
            if counters['status']!='complete' or counters['coverage']!={'blocks':338,'documents':500,'excluded_tail_tokens':1444} or counters['checkpoint']!=result['checkpoint'] or counters['gpu_uuid'] not in uuids:raise ValueError('Consumer counter identity/coverage')
            if counters['source']!=record(ROOT/counters['source']['path']):raise ValueError('Consumer diagnostic source identity')
            out['sources'].append(record(counter_path))
            out['conditions'][key]={'size':size,'pair':pair,'processes':processes,'modes':modes,'diagnostics':diagnostics,'consumer_counters':counters,'sparse_speedup_over_native':native/sparse,'sparse_reduction_vs_native_pct':100*(1-sparse/native),'sparse_speedup_over_no_skip':noskip/sparse,'sparse_reduction_vs_no_skip_pct':100*(1-sparse/noskip)}
    if len(uuids)!=1:raise ValueError('Physical GPU changed')
    out['gpu_uuid']=next(iter(uuids));out['logit_comparisons']=sum(p['logit_comparisons'] for c in out['conditions'].values() for p in c['processes'])
    write(ROOT/'results/decomposition.json',out)
    print({k:{'native_us':c['modes']['native']['baseline_host_us']['geomean'],'sparse_us':c['modes']['sparse']['baseline_host_us']['geomean'],'no_skip_us':c['modes']['no_skip']['baseline_host_us']['geomean']} for k,c in out['conditions'].items()})
    return out
if __name__=='__main__':main()

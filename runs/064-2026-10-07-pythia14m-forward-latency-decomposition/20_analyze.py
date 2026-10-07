"""Verify cohort and produce additive budgets without rescaling instrumented data."""
import collections
import hashlib
import json
import math
from pathlib import Path
import statistics
from accounting import budget,intervals_union
from trace_copy_mapping import map_native_copies

ROOT=Path(__file__).parent
def read(path):return json.loads(Path(path).read_text())
def lines(path):return [json.loads(line) for line in Path(path).read_text().splitlines()]
def describe(values):
    values=sorted(values);n=len(values)
    return {'count':n,'mean':statistics.mean(values),'geomean':math.exp(statistics.mean(map(math.log,values))) if all(v>0 for v in values) else None,
            'median':statistics.median(values),'p10':values[int((n-1)*.1)],'p90':values[int((n-1)*.9)],'min':min(values),'max':max(values)}
def category(name):return name.split('.',1)[-1]

def main():
    cfg=read(ROOT/'config.json');output={'status':'verified','question':'Full forward latency decomposition','pairs':{},'sources':[]}
    for pair in cfg['pairs']:
        processes=[];base=collections.defaultdict(list);stage_rows=collections.defaultdict(list);traces=[];internal=[]
        for replicate in range(1,4):
            dest=ROOT/'artifacts/attempts'/f'full-k{pair}-r{replicate}-003'
            result=read(dest/'result.json');quality=read(dest/'quality.json')
            if result['status']!='complete' or not result['qualified'] or quality['blocks']!=338 or not all(quality['pass'].values()):raise ValueError('Unqualified process '+str(dest))
            if quality['prediction_tokens']!=691886 or quality['documents']!=500 or quality['excluded_tail_tokens']!=1444:raise ValueError('Coverage')
            baseline=lines(dest/'baseline-samples.jsonl');stages=lines(dest/'stage-samples.jsonl')
            if len(baseline)!=64*7*5 or len(stages)!=len(baseline):raise ValueError('Incomplete timing coverage')
            comparisons=[row for rows in quality['gates'].values() for row in rows]
            if len(quality['gates'])!=10 or len(comparisons)!=338*10:raise ValueError('Incomplete numerical comparisons')
            process={'replicate':replicate,'loss':result['loss'],'gpu_uuid':result['runtime']['gpu_uuid'],'elapsed_seconds':result['elapsed_seconds'],
                'logit_comparisons':len(comparisons),'max_logit_absolute_error':max(r['max_abs'] for r in comparisons),
                'max_logit_relative_l2':max(r['relative_l2'] for r in comparisons),'peak_allocated_bytes':result['peak_allocated_bytes'],
                'peak_reserved_bytes':result['peak_reserved_bytes'],'setup_seconds':result['setup_seconds'],'timing':{}}
            for mode in cfg['modes']:
                rows=[r for r in baseline if r['mode']==mode];mapped=[r for r in stages if r['mode']==mode]
                base[mode].extend({**row,'process_replicate':replicate} for row in rows);stage_rows[mode].extend(mapped)
                process['timing'][mode]={'host_us':describe([r['host_ms']*1000 for r in rows]),'cuda_us':describe([r['cuda_ms']*1000 for r in rows])}
            processes.append(process)
            if replicate==1:
                traces=read(dest/'traces.json');internal=read(dest/'internal-cycles.json')
                if len(traces)!=len(cfg['modes'])*cfg['trace_inputs']:raise ValueError('Trace coverage')
                if len(internal)!=4*6*cfg['internal_inputs']:raise ValueError('Internal probe coverage')
                for row in internal:
                    if not row['exact_output'] or len(row['cycles'])!=cfg['internal_repeats']:raise ValueError('Internal probe fidelity/repeats')
                    if any(len(repeat)!=256 or any(len(block)!=4 or any(len(warp)!=4 or min(warp)<0 for warp in block) for block in repeat) for repeat in row['cycles']):raise ValueError('Internal clock coverage')
                diagnostics=[read(dest/f'diagnostics-{mode}.json') for mode in ['generic_sparse','tile_sparse']]
                if any(d['status']!='complete' or d['coverage']['blocks']!=338 for d in diagnostics):raise ValueError('Diagnostic coverage')
                if any(diagnostics[0][key]!=diagnostics[1][key] for key in diagnostics[0] if key!='elapsed_seconds'):raise ValueError('Consumer activation/counter mismatch')
            for path in sorted(dest.glob('*.json'))+sorted(dest.glob('*.jsonl')):
                output['sources'].append({'path':path.relative_to(ROOT).as_posix(),'bytes':path.stat().st_size,'sha256':hashlib.file_digest(path.open('rb'),'sha256').hexdigest()})
        if len({p['gpu_uuid'] for p in processes})!=1:raise ValueError('Physical GPU changed')
        modes={}
        for mode in cfg['modes']:
            names=stage_rows[mode][0]['stage_names']
            if any(r['stage_names']!=names for r in stage_rows[mode]):raise ValueError('Stage map changed')
            mapped=budget(names,[r['stage_us'] for r in stage_rows[mode]])
            categories=collections.defaultdict(float)
            for row in mapped['stages']:categories[category(row['stage'])]+=row['mean_us']
            baseline_host=describe([r['host_ms']*1000 for r in base[mode]]);baseline_cuda=describe([r['cuda_ms']*1000 for r in base[mode]])
            instrumented_host=describe([r['host_ms']*1000 for r in stage_rows[mode]])
            instrumented_cuda=describe([r['cuda_ms']*1000 for r in stage_rows[mode]])
            trace_stage=collections.defaultdict(list);trace_nodes={};trace_replays=[];signature_results=[]
            for trace in [t for t in traces if t['mode']==mode]:
                if trace['graph']['repeats']!=cfg['trace_replays']:raise ValueError('Trace replay coverage')
                mapping=map_native_copies(trace)
                signature_results.append(mapping['mapped'])
                if not mapping['mapped']:continue
                rows=mapping['kernels'];count=mapping['kernels_per_replay']
                for replay in range(trace['graph']['repeats']):
                    subset=rows[replay*count:(replay+1)*count];by_stage=collections.defaultdict(float)
                    for node_index,r in enumerate(subset):
                        by_stage[r['stage']]+=r['duration_us']
                        node=trace_nodes.setdefault(node_index,{'index':node_index,'stage':r['stage'],'kernel':r['name'],'samples_us':[]})
                        if node['stage']!=r['stage'] or node['kernel']!=r['name']:raise ValueError('Kernel-node signature changed across trace inputs')
                        node['samples_us'].append(r['duration_us'])
                    for name in names:trace_stage[name].append(by_stage[name])
                    active=intervals_union([(r['ts_us'],r['ts_us']+r['duration_us']) for r in subset])
                    span=max(r['ts_us']+r['duration_us'] for r in subset)-min(r['ts_us'] for r in subset)
                    trace_replays.append({'active_us':active,'span_us':span,'gap_us':span-active,'kernel_count':count,'summed_kernel_us':sum(r['duration_us'] for r in subset)})
            trace_budget={'all_signatures_match':all(signature_results),'mapped_inputs':sum(signature_results),'total_inputs':len(signature_results),'replays':trace_replays,
                          'mappings':[{'input_index':t['input_index'],'mapped':map_native_copies(t)['mapped'],'method':map_native_copies(t).get('mapping'),'copy_nodes':map_native_copies(t).get('graph_only_gate_copies',[])} for t in traces if t['mode']==mode],
                          'stages':[{'stage':name,**describe(v)} for name,v in trace_stage.items()]}
            trace_budget['kernel_nodes']=[{k:v for k,v in node.items() if k!='samples_us'}|describe(node['samples_us']) for node in trace_nodes.values()]
            if trace_replays:
                trace_budget.update(mean_active_us=statistics.mean(r['active_us'] for r in trace_replays),mean_span_us=statistics.mean(r['span_us'] for r in trace_replays),mean_gap_us=statistics.mean(r['gap_us'] for r in trace_replays))
                reference=[r['cuda_ms']*1000 for r in base[mode] if r['process_replicate']==1 and r['input_index']<cfg['trace_inputs']]
                trace_budget['matched_input_unprofiled_outer_cuda_us']=describe(reference)
                trace_budget['span_over_matched_outer_cuda_ratio']=trace_budget['mean_span_us']/statistics.mean(reference)
                trace_budget['ratio_boundary_caveat']='Profiled first-to-last activity span versus unprofiled outer-event bracket for the same4 inputs/process. Different boundaries; this is a consistency diagnostic, not exact CUPTI overhead.'
            internals=[r for r in internal if r['mode']==mode];cycle_summary=[]
            for layer in range(6):
                rows=[r for r in internals if r['layer']==layer]
                if not rows:continue
                values=[warp for r in rows for repeat in r['cycles'] for block in repeat for warp in block]
                stage_summary=[{'stage':name,**describe([v[i] for v in values])} for i,name in enumerate(rows[0]['stage_names'])]
                total=sum(r['mean'] for r in stage_summary)
                for r in stage_summary:r['fraction_of_mean_warp_cycles']=r['mean']/total if total else 0
                kernel_baseline=[p['base'] for r in rows for p in r['kernel_us']];kernel_clock=[p['clocked'] for r in rows for p in r['kernel_us']]
                cycle_summary.append({'layer':layer,'stages':stage_summary,'mean_total_warp_cycles':total,'kernel_base_us':describe(kernel_baseline),'kernel_clocked_us':describe(kernel_clock),
                    'probe_slowdown_mean_ratio':statistics.mean(kernel_clock)/statistics.mean(kernel_baseline),'input_density':[r['density'] for r in rows]})
            modes[mode]={'baseline_host_us':baseline_host,'baseline_cuda_us':baseline_cuda,'equal_staging_us':describe([r['staging_ms_excluded']*1000 for r in base[mode]]),
                'host_minus_cuda_mean_us':baseline_host['mean']-baseline_cuda['mean'],'instrumented_host_us':instrumented_host,'instrumented_cuda_us':instrumented_cuda,
                'event_instrumentation_cuda_mean_ratio':instrumented_cuda['mean']/baseline_cuda['mean'],'event_partition':mapped,'event_categories_us':dict(categories),
                'trace_budget':trace_budget,'internal_warp_cycles':cycle_summary}
        diagnostic=diagnostics[0]
        output['pairs'][pair]={'processes':processes,'modes':modes,
            'diagnostics':{'consumers_equal_except_elapsed_seconds':True,'coverage':diagnostic['coverage'],
                'hybrid_counter_fields':diagnostic['hybrid_counter_fields'],'hybrid_counts_by_layer':diagnostic['hybrid_counts_by_layer'],
                'hybrid_counts_pooled':[sum(row[i] for row in diagnostic['hybrid_counts_by_layer'].values()) for i in range(6)],
                'tiles8x16_by_site_layer':diagnostic['tiles8x16_by_site_layer']},
            'internal_probe_exact_outputs':len(internal),'internal_probe_total_warp_phase_intervals':len(internal)*cfg['internal_repeats']*256*4*4}
        if any(not row['trace_budget']['all_signatures_match'] for row in modes.values()):raise ValueError('Full stage mapping remains unresolved')
    if len({p['gpu_uuid'] for pair in output['pairs'].values() for p in pair['processes']})!=1:raise ValueError('Cohort GPU changed')
    (ROOT/'results').mkdir(exist_ok=True)
    (ROOT/'results/decomposition.json').write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')
    print(json.dumps({pair:{mode:row['baseline_host_us']['geomean'] for mode,row in value['modes'].items()} for pair,value in output['pairs'].items()},indent=2))
    return output
if __name__=='__main__':main()

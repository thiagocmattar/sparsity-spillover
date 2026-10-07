"""Run-owned comprehensive tables and publication PDF diagnostics."""
import collections
import csv
import json
from pathlib import Path
import statistics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

ROOT=Path(__file__).parent
data=json.loads((ROOT/'results/decomposition.json').read_text())
MODES=['native','generic_sparse','tile_sparse','generic_no_skip','tile_no_skip']
LABELS={'native':'Native','generic_sparse':'Generic sparse','tile_sparse':'Specialized sparse','generic_no_skip':'Generic no skip','tile_no_skip':'Specialized no skip'}
CATEGORIES=['positions','embedding','rotary_tables','norm_pair','qkv','rope_qkv_layout','attention','context_pack','mlp_up','joint_gate_down_residual','final_norm','lm_head']
FRIENDLY={'positions':'Position indices','embedding':'Embedding','rotary_tables':'Rotary tables','norm_pair':'Paired normalization','qkv':'QKV projection','rope_qkv_layout':'RoPE / QKV layout','attention':'Causal attention','context_pack':'Context packing','mlp_up':'MLP up projection','joint_gate_down_residual':'Gates / down projections / residual','final_norm':'Final normalization','lm_head':'Vocabulary projection'}
NATIVE_REMAP={'norm_attention':'norm_pair','norm_mlp':'norm_pair','z_gate':'joint_gate_down_residual','h_gate':'joint_gate_down_residual','attention_out':'joint_gate_down_residual','mlp_down':'joint_gate_down_residual','residual':'joint_gate_down_residual'}
def grouped(rows,value):
    out=collections.defaultdict(float)
    for row in rows:
        category=row['stage'].split('.',1)[-1];category=NATIVE_REMAP.get(category,category)
        out[category]+=row[value]
    return out
def fmt(value):return f'{value:.3f}'

def main():
    observations=ROOT/'observations';observations.mkdir(exist_ok=True)
    figures=ROOT/'figures';figures.mkdir(exist_ok=True)
    with (ROOT/'results/kernel-nodes.csv').open('w',newline='',encoding='utf-8') as handle:
        fields=['kappa','mode','index','stage','kernel','count','mean','geomean','median','p10','p90','min','max']
        writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader()
        for pair in ['0','0p1']:
            for mode in MODES:
                for row in data['pairs'][pair]['modes'][mode]['trace_budget']['kernel_nodes']:
                    writer.writerow({'kappa':pair.replace('p','.'),'mode':mode,**row})
    report=['# Full forward inference latency distribution','','Question: where is full-sequence inference time spent, and which stages explain the generic-to-specialized difference?','',
            'Method: one physical RTX 5090; BF16 B=1, T=2048, uncached full 50,304-way logits. Same retained Run060 T2/Ph checkpoints and corner 8x16 gate at kappa 0/.1. Three fresh processes per threshold, 64 fixed validation inputs x 7 randomized paired passes. Full 500-document/338-block qualification covers every baseline and mapped graph; 1,444-token excluded tail. No weights, thresholds, arithmetic acceptance bounds or scientific kernel settings changed.','',
            'The full step means a forward for all 2,048 tokens, not one decoding token. Input copies are measured separately; dataset reads, model loading, weight preparation, compilation, allocation and capture are setup. Cross-entropy/loss and diagnostics are qualification work outside the inference timer.','',
            '## Detailed execution map','',
            '| Order | Mathematical work / shapes | Actual optimized execution |','|---|---|---|',
            '| 1 | Position indices [1,2048] | arange kernel; view is metadata |',
            '| 2 | Token lookup [1,2048,128] | embedding lookup; eval dropout identity |',
            '| 3 | Shared partial RoPE cos/sin tables | recurring table generation for this forward |',
            '| 4a, each of 6 layers | Two LayerNorms of the same residual, distinct affine weights | one K050 Welford normalization kernel; MLP retrieves the second output later |',
            '| 4b | QKV: [2048,128] x [128,384], including bias | K042 CUTLASS projection; no a gate |',
            '| 4c | Q/K partial RoPE and V packing; [1,4,2048,32] | K019 fused RoPE/layout; identity Q/K/V taps |',
            '| 4d | Causal QK, scaled softmax, PV | frozen K035 split-KV attention and split recombination kernels; QK/softmax/PV are fused, with no independently measured wall times |',
            '| 4e | Heads -> [1,2048,128] context | transpose view plus contiguous packing copy; original z gate is deferred to joint |',
            '| 4f | MLP up: [2048,128] x [128,512], bias | K042 CUTLASS projection; no m gate and h gate deferred |',
            '| 4g | h/z tile gate, W2 [512,128], Wo [128,128], bias, rounded branch sum and residual add | one joint CUDA kernel; generic or corner-specialized; SAME mathematical outputs |',
            '| 5 | Final LayerNorm [1,2048,128] | native affine normalization |',
            '| 6 | Vocabulary head [2048,128] x [128,50304] | dense cuBLAS projection, [1,2048,50304] logits |','',
            'GPT-NeoX has parallel mathematical branches; this single-stream implementation executes the attention work then MLP work before joining them. LayerNorm pairs and joint projections are fusions, not concurrent whole-model branches. Metadata views, identity taps and eval dropout have no independent CUDA kernel. Their host work appears in annotated eager CPU traces.','',
            '## Full forward, overhead and measurement coverage','',
            '| kappa | Mode | Host geom mean (us) | Host arithmetic mean (us) | CUDA arithmetic mean (us) | Host minus CUDA mean (us) | Timed-event graph / baseline CUDA |','|---|---|---:|---:|---:|---:|---:|']
    conclusions=['## Measured result','',
        'Accepted evidence is the six complete fresh processes in infrastructure attempt 003. The two earlier infrastructure failures are preserved but excluded. All 20,280 full-block logit comparisons have zero absolute and relative-L2 error; losses are exactly 5.556248486078721 and 5.5987847543876 at kappa 0/.1. The generic and specialized consumers also have identical full-validation activation, tile and MMA/SIMT counters.','']
    for pair in ['0','0p1']:
        modes=data['pairs'][pair]['modes'];g=modes['generic_sparse'];s=modes['tile_sparse']
        gc=grouped(g['trace_budget']['stages'],'mean');sc=grouped(s['trace_budget']['stages'],'mean')
        reduction=100*(1-s['baseline_host_us']['geomean']/g['baseline_host_us']['geomean'])
        penalty=100*(modes['tile_no_skip']['baseline_host_us']['geomean']/modes['generic_no_skip']['baseline_host_us']['geomean']-1)
        active=s['trace_budget']['mean_active_us']
        conclusions.append(f'At kappa {pair.replace("p",".")}, synchronized host latency falls {g["baseline_host_us"]["geomean"]:.3f} to {s["baseline_host_us"]["geomean"]:.3f} us ({reduction:.3f}%). The six joint gate/down-projection/residual kernels fall {gc["joint_gate_down_residual"]:.3f} to {sc["joint_gate_down_residual"]:.3f} us in the CUPTI sample, a {gc["joint_gate_down_residual"]-sc["joint_gate_down_residual"]:.3f}-us saving. Total kernel-active time falls {g["trace_budget"]["mean_active_us"]-active:.3f} us: the changed stage accounts for essentially all of that observed difference, with small offsetting variation elsewhere. The remaining specialized active budget is {100*sc["lm_head"]/active:.2f}% vocabulary projection, {100*sc["attention"]/active:.2f}% attention, and {100*sc["joint_gate_down_residual"]/active:.2f}% joint kernels. With skipping disabled, specialization is {penalty:.3f}% slower.')
        conclusions.append('')
    ratios=[r['event_instrumentation_cuda_mean_ratio'] for p in data['pairs'].values() for r in p['modes'].values()]
    conclusions += [f'The event-rich mapped graphs add {100*(min(ratios)-1):.1f}% to {100*(max(ratios)-1):.1f}% to mean CUDA latency. Their detailed partitions are useful for the execution map, but the less intrusive CUPTI kernel durations are the primary stage-localization evidence. The report keeps baseline wall time, profiled active time/gaps, instrumented intervals and SM cycles as separate budgets.','',
        'The specialized source reads one corner per 8x16 tile, removes the generic full inspection/shared-memory/barrier path, and launches 128 rather than 256 threads while retaining four projection warps. Timestamp probes show much shorter support/preparation and accumulation intervals, consistent with removing support work and changing warp scheduling. The preparation boundaries differ and clocks include stalls; these measurements do not identify a causal percentage for mask inspection, occupancy, memory traffic or arithmetic. Identical MMA counters rule out additional product skipping as the generic-to-specialized explanation.','']
    report[4:4]=conclusions
    for pair in ['0','0p1']:
        for mode in MODES:
            r=data['pairs'][pair]['modes'][mode]
            report.append(f'| {pair.replace("p",".")} | {LABELS[mode]} | {fmt(r["baseline_host_us"]["geomean"])} | {fmt(r["baseline_host_us"]["mean"])} | {fmt(r["baseline_cuda_us"]["mean"])} | {fmt(r["host_minus_cuda_mean_us"])} | {r["event_instrumentation_cuda_mean_ratio"]:.4f}x |')
    report+=['','Host minus CUDA is a timer-boundary residual including launch/synchronization/timing overhead, not a precisely isolated CPU compute stage. Event instrumentation includes extra graph event nodes and may alter scheduling/cache state. Event stage budgets close to their OWN instrumented totals and are never rescaled to uninstrumented latency. Arithmetic means, not geometric means or summed medians, define additive budgets.','',
        '| kappa | Process replicate | Generic sparse host geom (us) | Specialized sparse host geom (us) | Reduction (%) |','|---|---:|---:|---:|---:|']
    for pair in ['0','0p1']:
        for p in data['pairs'][pair]['processes']:
            g=p['timing']['generic_sparse']['host_us']['geomean'];s=p['timing']['tile_sparse']['host_us']['geomean']
            report.append(f'| {pair.replace("p",".")} | {p["replicate"]} | {g:.3f} | {s:.3f} | {100*(1-s/g):.3f} |')
    report+=['','Equal staging is a device-to-device copy into the graph input buffer, synchronization and outer event construction. It is timed separately from resident-input inference; disk reads and initial CPU-to-GPU dataset staging are setup. Annotated eager CPU stages are retained in raw traces, but steady graph replay dispatches one graph rather than executing those Python stages.','',
        '| kappa | Mode | Staging host arithmetic mean (us), excluded |','|---|---|---:|']
    for pair in ['0','0p1']:
        for mode in MODES:report.append(f'| {pair.replace("p",".")} | {LABELS[mode]} | {data["pairs"][pair]["modes"][mode]["equal_staging_us"]["mean"]:.3f} |')
    with PdfPages(figures/'01-forward-latency-decomposition.pdf') as pdf:
        colors=plt.get_cmap('tab20').colors[:len(CATEGORIES)]
        for pair in ['0','0p1']:
            modes=data['pairs'][pair]['modes']
            report += ['',f'## kappa {pair.replace("p",".")}: per-stage distribution','',
                'Baseline graph CUPTI activity time, without internal timing-event nodes; arithmetic mean over 4 inputs x 3 replays. CUPTI profiling remains active. Sum covers measured GPU activities; inter-kernel gaps and host overhead are separate. Computational kernels require exact ordered kernel-name agreement with the annotated equivalent eager map. Native replay also has 12 graph copy activities, one in each h/z gate: eager records these as device-to-device memcpy, while graph replay records memcpy32_post/memcpy128 kernels. The local mapper accepts them only when enclosed by kernels of the same gate, with exactly one copy per gate/layer and identical signatures in every replay. The original strict-mapping failures remain in the raw evidence. This is a 12-forward profiling sample, separate from the 1,344 baseline samples per mode.','',
                '| Stage, pooled over 6 layers where applicable | Native (us) | Generic sparse (us) | Specialized sparse (us) | Generic no skip (us) | Specialized no skip (us) |','|---|---:|---:|---:|---:|---:|']
            trace_groups={mode:grouped(modes[mode]['trace_budget']['stages'],'mean') for mode in MODES}
            all_mapped=all(modes[mode]['trace_budget']['all_signatures_match'] for mode in MODES)
            if all_mapped:
                for category in CATEGORIES:report.append('| '+FRIENDLY[category]+' | '+' | '.join(fmt(trace_groups[mode][category]) for mode in MODES)+' |')
                report.append('| Total kernel-active time | '+' | '.join(fmt(modes[mode]['trace_budget']['mean_active_us']) for mode in MODES)+' |')
                report+=['','Specialized sparse distribution, using its own profiled GPU-active denominator:','',
                    '| Stage | Mean (us) | Active share (%) | Generic minus specialized (us) |','|---|---:|---:|---:|']
                for category in CATEGORIES:
                    value=trace_groups['tile_sparse'][category]
                    report.append(f'| {FRIENDLY[category]} | {value:.3f} | {100*value/modes["tile_sparse"]["trace_budget"]["mean_active_us"]:.2f} | {trace_groups["generic_sparse"][category]-value:.3f} |')
            else:report.append('Kernel sequence mapping incomplete: consult results/decomposition.json and raw traces. Unmapped values are not claimed.')
            report+=['','| Mode | Mapped inputs | Activities/forward | Active union (us) | First-to-last activity span (us) | Gaps in span (us) | Profile span / matched unprofiled CUDA bracket |','|---|---:|---:|---:|---:|---:|---:|']
            for mode in MODES:
                t=modes[mode]['trace_budget'];replays=t['replays']
                report.append(f'| {LABELS[mode]} | {t["mapped_inputs"]}/{t["total_inputs"]} | {replays[0]["kernel_count"] if replays else "unmapped"} | {fmt(t.get("mean_active_us",0))} | {fmt(t.get("mean_span_us",0))} | {fmt(t.get("mean_gap_us",0))} | {t.get("span_over_matched_outer_cuda_ratio",0):.4f}x |')
            report+=['','The final ratio compares the profiled first-to-last activity span with unprofiled outer-CUDA-event timings of the same 4 inputs in the same process. Their timer boundaries differ, so the ratio is a consistency check, not an exact estimate of CUPTI overhead. No latency budget mixes these samples.']
            report+=['','Inside attention, the actually separate CUDA kernels can be timed. The fused split-attention kernel includes QK, online softmax and PV; the second kernel combines splits. Native attention also includes a layout copy in this stage.','',
                '| Attention component, pooled over 6 layers | Native (us) | Generic sparse (us) | Specialized sparse (us) | Generic no skip (us) | Specialized no skip (us) |','|---|---:|---:|---:|---:|---:|']
            attention_parts={}
            for mode in MODES:
                parts=collections.defaultdict(float)
                for row in modes[mode]['trace_budget']['kernel_nodes']:
                    if row['stage'].split('.',1)[-1]!='attention':continue
                    role='Split recombination' if 'combine' in row['kernel'] else 'Fused QK / softmax / PV' if ('splitkv_kernel' in row['kernel'] or 'run028_k035_flash::kernel' in row['kernel']) else 'Layout copy'
                    parts[role]+=row['mean']
                attention_parts[mode]=parts
            for role in ['Fused QK / softmax / PV','Split recombination','Layout copy']:
                report.append('| '+role+' | '+' | '.join(f'{attention_parts[mode][role]:.3f}' for mode in MODES)+' |')
            report+=['','Rotary-table generation also comprises separate small CUDA operations. The specialized sparse graph calls, in execution order:','',
                '| Rotary table substep | Mean (us) |','|---|---:|']
            rotary=[r for r in modes['tile_sparse']['trace_budget']['kernel_nodes'] if r['stage']=='rotary_tables']
            rotary_labels=['Inverse-frequency dtype copy','Position dtype copy','Frequency-position product','Frequency concatenation','Cosine','Cosine scaling','Sine','Sine scaling','Cosine BF16 cast','Sine BF16 cast']
            rotary_signatures=['direct_copy','direct_copy','gemmk1_kernel','CatArrayBatchedCopy','cos_kernel_cuda','MulFunctor','sin_kernel_cuda','MulFunctor','bfloat16_copy','bfloat16_copy']
            if len(rotary)!=len(rotary_labels) or any(sig not in row['kernel'] for sig,row in zip(rotary_signatures,rotary)):raise ValueError('Rotary execution signature changed')
            for label,row in zip(rotary_labels,rotary):report.append(f'| {label} | {row["mean"]:.3f} |')
            report+=['','[Complete kernel-node inventory](../results/kernel-nodes.csv) lists every ordered CUDA activity, full kernel name, mapped stage and duration distribution for all five modes and both thresholds. Raw timelines retain their timestamps.']
            report+=['','Every optimized execution stage in the CUPTI trace, separately for all six layers:','',
                '| Stage | Generic sparse mean (us) | Specialized sparse mean (us) | Difference (us) |','|---|---:|---:|---:|']
            g={r['stage']:r['mean'] for r in modes['generic_sparse']['trace_budget']['stages']}
            s={r['stage']:r['mean'] for r in modes['tile_sparse']['trace_budget']['stages']}
            for name in g:report.append(f'| {name} | {g[name]:.3f} | {s[name]:.3f} | {g[name]-s[name]:.3f} |')
            report+=['','Native graph operations before fusion, pooled over six layers:','',
                '| Native operation | Mean GPU activity (us) |','|---|---:|']
            raw_native=collections.defaultdict(float)
            for row in modes['native']['trace_budget']['stages']:raw_native[row['stage'].split('.',1)[-1]]+=row['mean']
            for name,value in raw_native.items():report.append(f'| {name} | {value:.3f} |')
            diagnostic=data['pairs'][pair]['diagnostics']
            report+=['','Full-validation projection instruction counters are identical between consumers. These count declared MMA instructions, not hardware-measured FLOPs or memory traffic.','',
                '| Counter | Pooled integer count, both consumers |','|---|---:|']
            for name,value in zip(diagnostic['hybrid_counter_fields'],diagnostic['hybrid_counts_pooled']):report.append(f'| {name} | {value:,} |')
            report+=['','Instrumented graph event budget: all 64 inputs x 7 passes x 3 processes. These intervals include event-node/scheduling costs and close to the instrumented forward.','',
                    '| Execution stage | Generic sparse mean (us) | Specialized sparse mean (us) | Difference (us) |','|---|---:|---:|---:|']
            event_groups={mode:grouped(modes[mode]['event_partition']['stages'],'mean_us') for mode in MODES}
            for category in CATEGORIES:
                g=event_groups['generic_sparse'][category];s=event_groups['tile_sparse'][category]
                report.append(f'| {FRIENDLY[category]} | {fmt(g)} | {fmt(s)} | {fmt(g-s)} |')
            report+=['','Complete per-layer event measurements:','',
                    '| Stage | Generic sparse mean (us) | Specialized sparse mean (us) |','|---|---:|---:|']
            g={r['stage']:r['mean_us'] for r in modes['generic_sparse']['event_partition']['stages']}
            s={r['stage']:r['mean_us'] for r in modes['tile_sparse']['event_partition']['stages']}
            for name in g:report.append(f'| {name} | {fmt(g[name])} | {fmt(s[name])} |')
            fig,axes=plt.subplots(2,1,figsize=(12,9),gridspec_kw={'height_ratios':[2,1.2]})
            groups=trace_groups if all_mapped else event_groups
            left=[0.]*len(MODES)
            for category,color in zip(CATEGORIES,colors):
                values=[groups[mode][category] for mode in MODES]
                axes[0].barh(range(len(MODES)),values,left=left,color=color,label=FRIENDLY[category]);left=[a+b for a,b in zip(left,values)]
            axes[0].set_yticks(range(len(MODES)),[LABELS[m] for m in MODES]);axes[0].invert_yaxis()
            axes[0].set_xlabel('Mean GPU kernel-active time (microseconds)' if all_mapped else 'Instrumented event intervals (microseconds)')
            axes[0].set_title('Full 2048-token forward latency distribution; kappa='+pair.replace('p','.'))
            axes[0].legend(loc='upper center',bbox_to_anchor=(.5,-.16),ncol=3,fontsize=8,frameon=False)
            layers=range(6)
            for mode,color in [('generic_sparse','#2878b5'),('tile_sparse','#ed8b23')]:
                source={r['stage']:r['mean'] for r in modes[mode]['trace_budget']['stages']} if all_mapped else {r['stage']:r['mean_us'] for r in modes[mode]['event_partition']['stages']}
                axes[1].plot(layers,[source[f'layer{i}.joint_gate_down_residual'] for i in layers],marker='o',label=LABELS[mode],color=color)
            axes[1].set_xticks(list(layers));axes[1].set_xlabel('Transformer layer');axes[1].set_ylabel('Joint kernel mean (us)');axes[1].legend(frameon=False)
            fig.subplots_adjust(left=.19,right=.97,top=.93,bottom=.08,hspace=.85);pdf.savefig(fig);plt.close(fig)
            report+=['','Internal timestamp diagnostic (mean SM cycles per projection warp; every 8-token block, 8 inputs x 5 repeats). Support/preparation, h, z and output phases are serial within a warp but overlap across warps/CTAs. They cannot be added to full-device latency. The generic preparation phase includes inspection, a block barrier and short/zero-row bias/output handling; the specialized phase reads corners. For a wholly dead generic block, its bias/residual output happens in preparation, and later phases are zero. These phase labels therefore do not isolate pure mask cost or provide clean causal percentages. Probe slowdown is independently measured and retained.','',
                    '| Mode / layer | Support cycles | h accumulation cycles | z accumulation cycles | Output cycles | Probe / production event time |','|---|---:|---:|---:|---:|---:|']
            fig,axes=plt.subplots(2,2,figsize=(12,8),sharex=True,sharey=True)
            for mode,ax in zip(MODES[1:],axes.flat):
                rows=modes[mode]['internal_warp_cycles'];bottom=[0.]*6
                for j,label in enumerate(['Support / preparation','h accumulation','z accumulation','Rounded output / residual']):
                    values=[r['stages'][j]['mean'] for r in rows];ax.bar(range(6),values,bottom=bottom,label=label);bottom=[a+b for a,b in zip(bottom,values)]
                ax.set_title(LABELS[mode]);ax.set_xticks(range(6));ax.set_xlabel('Layer');ax.set_ylabel('Mean SM cycles per warp')
                for row in rows:report.append(f'| {LABELS[mode]} / {row["layer"]} | '+' | '.join(f'{r["mean"]:.1f}' for r in row['stages'])+f' | {row["probe_slowdown_mean_ratio"]:.3f}x |')
            axes[0,0].legend(fontsize=8,frameon=False);fig.suptitle('Diagnostic timestamp kernel copies; kappa='+pair.replace('p','.')+' (cycles are not device wall time)')
            fig.tight_layout(rect=(0,0,1,.95));pdf.savefig(fig);plt.close(fig)
    report+=['','## Interpretation limits and provenance','',
             'The no-skip probes illustrate the boundary problem: specialized projection warps have shorter clock intervals while the full specialized no-skip kernel/forward is slower. Warp-local elapsed time omits the whole-device distribution of resident and queued blocks. The cycle partitions must not be used as device latency or as a hardware occupancy measurement.','',
             'One trained seed, two thresholds, one physical GPU and one uncached full-sequence shape. CUPTI sample size and measurement perturbation are explicit. No individual full-device timings are invented for operations fused into a kernel. Internal SM clocks include warp scheduling stalls and instrumentation, and do not measure DRAM bytes, hardware FLOPs, pure arithmetic cost or causal percentages of the speedup. No kernel or model tuning was performed. The original trained tile-model quality penalty remains.','',
             'Every raw kernel timeline, full numerical comparison, paired timing sample, per-layer stage interval, internal cycle/density sample, activation/weight diagnostic, setup/compiler log and transfer inventory is retained under artifacts/. The reducer rejects incomplete timing/full validation coverage and different GPU UUIDs.','',
             'Source scripts: [20_analyze.py](../20_analyze.py), [21_report.py](../21_report.py), [stages.py](../stages.py), [profiling.py](../profiling.py), [internal.py](../internal.py), [native copy mapper](../trace_copy_mapping.py).','Evidence: [decomposition.json](../results/decomposition.json) and [complete kernel-node inventory](../results/kernel-nodes.csv). Figure caption: full forward distributions and per-layer joint timings, followed by diagnostic warp-cycle partitions; units and sampling follow the tables above.','Figure: [four-page PDF](../figures/01-forward-latency-decomposition.pdf).']
    (observations/'01-forward-decomposition.md').write_text('\n'.join(report)+'\n')
    (observations/'INDEX.md').write_text('# Observations\n\n- [Full forward decomposition](01-forward-decomposition.md): execution map, latency distributions, fusion limits and profiling overhead.\n')
    print('Comprehensive observation and four-page PDF generated')
if __name__=='__main__':main()

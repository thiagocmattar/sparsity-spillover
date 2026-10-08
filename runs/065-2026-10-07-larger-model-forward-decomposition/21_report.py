"""Run-owned complete timing tables and publication PDF from verified sources."""
import collections
import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from runio import ROOT,read,write,record

LABELS={'positions':'Positions','embedding':'Embedding','rotary_tables':'RoPE tables','norm_pair':'Layer norms','norm_attention':'Layer norms','norm_mlp':'Layer norms','qkv':'QKV projection','rope_qkv_layout':'RoPE / QKV layout','attention':'Attention','context_pack':'Context layout','mlp_up':'MLP up projection','h_gate':'Scalar gates','z_gate':'Scalar gates','attention_out':'Z down projection (+ fused gate)','z_gate_dense_down':'Z down projection (+ fused gate)','h_dense_down':'H down projection','h_gathered_down':'H down projection','mlp_down':'H down projection','h_feature_compaction':'H feature compaction','residual':'Residual sum','joint_gate_down_residual':'Joint gates / down / residual','final_norm':'Final norm','lm_head':'Vocabulary projection'}
ORDER=list(dict.fromkeys(LABELS.values()))
def aggregates(mode):
    values=collections.defaultdict(float)
    for r in mode['trace_budget']['stages']:values[LABELS[r['stage'].split('.',1)[-1]]]+=r['mean']
    return dict(values)
def branch_us(mode):
    names={'Scalar gates','Z down projection (+ fused gate)','H down projection','H feature compaction','Residual sum','Joint gates / down / residual'}
    return sum(value for label,value in aggregates(mode).items() if label in names)
def table(headers,rows):return '| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+''.join('| '+' | '.join(map(str,row))+' |\n' for row in rows)
def main():
    data=read(ROOT/'results/decomposition.json');dest=ROOT/'observations';dest.mkdir(exist_ok=True)
    modes=['native','sparse','no_skip'];stagecsv=[];kernelcsv=[];categorycsv=[];summary=[];text=[]
    for key,c in data['conditions'].items():
        summary.append([c['size'],c['pair'].replace('0p1','0.1')]+[f"{c['modes'][m]['baseline_host_us']['geomean']/1000:.4f}" for m in modes]+[f"{c['sparse_speedup_over_native']:.3f}x",f"{c['sparse_reduction_vs_no_skip_pct']:.2f}%"])
        for m,row in c['modes'].items():
            events={r['stage']:r for r in row['event_partition']['stages']}
            for stage in row['trace_budget']['stages']:
                stagecsv.append({'condition':key,'mode':m,'stage':stage['stage'],'category':LABELS[stage['stage'].split('.',1)[-1]],'cupti_mean_us':stage['mean'],'cupti_p10_us':stage['p10'],'cupti_p90_us':stage['p90'],'active_fraction':stage['fraction_of_active'],'event_mean_us':events[stage['stage']]['mean_us']})
            for node in row['trace_budget']['kernel_nodes']:kernelcsv.append({'condition':key,'mode':m,**node})
            for label,value in aggregates(row).items():categorycsv.append({'condition':key,'mode':m,'category':label,'cupti_mean_us':value,'active_fraction':value/row['trace_budget']['mean_active_us']})
    for name,rows in [('forward-stages.csv',stagecsv),('cuda-kernel-nodes.csv',kernelcsv),('forward-categories.csv',categorycsv)]:
        with (ROOT/'results'/name).open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    prior_path=ROOT.parent/'064-2026-10-07-pythia14m-forward-latency-decomposition/results/decomposition.json'
    prior=read(prior_path);cross=[]
    for pair,c in prior['pairs'].items():
        m=c['modes']['tile_sparse'];active=m['trace_budget']['mean_active_us'];stages=m['trace_budget']['stages']
        head=sum(r['mean'] for r in stages if r['stage']=='lm_head');attention=sum(r['mean'] for r in stages if r['stage'].endswith('.attention'))
        cross.append(['14M corner tiles',pair.replace('0p1','0.1'),f'{m["baseline_host_us"]["geomean"]/1000:.4f}',f'{head/active*100:.2f}%',f'{attention/active*100:.2f}%'])
    for c in data['conditions'].values():
        m=c['modes']['sparse'];cats=aggregates(m);active=m['trace_budget']['mean_active_us']
        cross.append([c['size'].upper()+' scalar',c['pair'].replace('0p1','0.1'),f'{m["baseline_host_us"]["geomean"]/1000:.4f}',f'{cats["Vocabulary projection"]/active*100:.2f}%',f'{cats["Attention"]/active*100:.2f}%'])
    text+=['# Existing 31M/70M full-forward latency decomposition','',
      'Question: does the latency distribution and sparse benefit persist in the retained larger T2/Ph models?','',
      'Method/coverage: one RTX5090, BF16 batch1,2048 tokens, uncached full50304-vocabulary output. Four final712-update checkpoints; three fresh processes each;64 matched random inputs ×7 randomized paired passes. Every baseline and mapped graph qualified against its own native eager model on all338 complete validation blocks from500 documents;1444 tail tokens excluded. Exact checkpoint/cache/source hashes are retained. No training, weights, threshold, gate or kernel policy was changed.','',
      'The31M frozen consumer is Run054 opt073-31m-v1;70M is Run051 sparse_c. Both consume elementwise one-sided h/z thresholds. Run06414M consumes corner-selected8x16 tiles, so the cross-size comparison is descriptive and includes differences in gate and implementation. It cannot establish architecture-only scaling of the new14M corner kernel.','',
      'Uninstrumented complete-forward synchronized host latency, geometric means in milliseconds. Controls retain the same gates and non-sparse components. The70M no-skip control also retains compaction, then computes the full K range.','',
      table(['Model','κ','Native ms','Sparse ms','No-skip ms','Native / sparse','Sparse reduction vs no-skip'],summary),'',
      'Result: the distribution changes with size and implementation.31M full latency falls4.63%/15.61% versus native, but its fused joint stage takes431.56/315.84µs versus172.88/172.85µs for native gates/down/residual. Its other component and dispatch savings compensate for that slower joint consumer. At κ0 the short-row prepass completes no token row in the entire338-block workload; at κ.1 it helps mainly layers1–3.70M full latency falls28.22%/31.03%; its five gathered h consumers benefit strongly from skipping, and vocabulary projection becomes its largest component at36.55%/38.01% of sparse GPU-active time. This does not reproduce a uniform cross-size bottleneck or demonstrate the new14M corner consumer at larger sizes.','',
      table(['Model / κ','Native gates/down/residual µs','Sparse µs','No-skip µs'],[[key]+[f'{branch_us(c["modes"][m]):.3f}' for m in modes] for key,c in data['conditions'].items()]),'',
      'Descriptive context against Run06414M (a different physical RTX5090 session and a different gate/consumer). These rows compare observed bottlenecks, not a matched architecture-only effect:','',
      table(['Model / gate','κ','Sparse host ms','Vocabulary active share','Attention active share'],cross),'',
      'Forward map: position indices → token embedding → RoPE tables → six transformer blocks with parallel attention/MLP residual topology → final norm → vocabulary projection. Each block computes attention and MLP from the same residual input; these measurements execute the GPU stages serially on one stream. Native: attention norm → QKV → rotary/layout → attention → context layout → z gate → z down; MLP norm → MLP up → h gate → h down; then both branches + residual. Optimized paired norm computes both norm outputs once.31M combines scalar inspection/short-row handling, h/z down projections, BF16 branch rounding and residual into the short-row prepass and joint fallback.70M exposes h0 gate + dense linear, h1..5 feature compaction + gathered linear, z fused gate + dense dot, and BF16 residual combination. Context layout may have zero CUDA activities when it is a view.','',
      'Primary component numbers below are summed actual CUPTI activity durations. Four timing inputs/process1 ×3 graph replays per mode; all computational kernel names and launch order must match the annotated eager map. The narrowly declared native graph-only memcpy mapping retains the original strict failure and attributes only enclosed gate copy nodes. GPU active time is an interval union; gaps are measured separately. CUDA-event partitions are additive but perturb execution; their overhead is reported and never rescaled into the baseline. Timing distributions, every layer/stage and every ordered CUDA activity are in the CSVs.','']
    figures=ROOT/'figures';figures.mkdir(exist_ok=True)
    with PdfPages(figures/'01-larger-forward-decomposition.pdf') as pdf:
        colors=['#0072B2','#E69F00','#009E73','#D55E00','#CC79A7','#56B4E9','#F0E442','#999999']
        fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
        for ax,(key,c) in zip(axes.flat,data['conditions'].items()):
            left=[0.]*3
            for i,label in enumerate(ORDER):
                vals=[aggregates(c['modes'][m]).get(label,0) for m in modes]
                if max(vals)==0:continue
                ax.barh(modes,vals,left=left,label=label,color=colors[i%8],hatch=['','///','...'][i//8],edgecolor='white',linewidth=.3);left=[a+b for a,b in zip(left,vals)]
            ax.set_title(key.replace('0p1','0.1'));ax.set_xlabel('Measured GPU activity per forward (µs)');ax.invert_yaxis();ax.grid(axis='x',alpha=.2)
        handles,labels=axes.flat[0].get_legend_handles_labels()
        # Include70M-specific components absent in the31M legend.
        all_legend={label:handle for ax in axes.flat for handle,label in zip(*ax.get_legend_handles_labels())}
        fig.legend(all_legend.values(),all_legend.keys(),loc='outside lower center',ncol=4,fontsize=7)
        upper=max(sum(aggregates(c['modes'][m]).values()) for c in data['conditions'].values() for m in modes)
        for ax in axes.flat:ax.set_xlim(0,upper*1.03)
        fig.suptitle('Retained scalar-threshold 31M/70M models — complete forward GPU decomposition')
        pdf.savefig(fig,bbox_inches='tight',pad_inches=.18);plt.close(fig)
        for key,c in data['conditions'].items():
            rows=c['modes'];cats={m:aggregates(rows[m]) for m in modes};active=rows['sparse']['trace_budget']['mean_active_us']
            text+=['## '+key.replace('0p1','0.1'),'',
              table(['Step category','Native µs','Sparse µs','No-skip µs','Sparse active share'],[[label]+[f'{cats[m][label]:.3f}' if label in cats[m] else 'N/A' for m in modes]+[f'{cats["sparse"][label]/active*100:.2f}%' if label in cats['sparse'] else 'N/A'] for label in ORDER if any(cats[m].get(label,0)>0 for m in modes)]),'',
              table(['Accounting','Native','Sparse','No-skip'],[
                ['GPU active µs']+[f'{rows[m]["trace_budget"]["mean_active_us"]:.3f}' for m in modes],
                ['First-to-last activity span µs']+[f'{rows[m]["trace_budget"]["mean_span_us"]:.3f}' for m in modes],
                ['Inter-activity gaps µs']+[f'{rows[m]["trace_budget"]["mean_gap_us"]:.3f}' for m in modes],
                ['Host minus outer CUDA mean µs']+[f'{rows[m]["host_minus_cuda_mean_us"]:.3f}' for m in modes],
                ['Equal input staging mean µs (excluded)']+[f'{rows[m]["equal_staging_us"]["mean"]:.3f}' for m in modes],
                ['Event instrumentation CUDA slowdown']+[f'{rows[m]["event_instrumentation_cuda_mean_ratio"]:.3f}x' for m in modes],
                ['Activities per graph']+[rows[m]['trace_budget']['replays'][0]['kernel_count'] for m in modes]]),'']
            text+=['Fresh-process host geometric means and pooled sample distributions (µs):','',table(['Mode','Process1','Process2','Process3','Pooled median','Pooled p10','Pooled p90'],[[m]+[f'{p["timing"][m]["host_us"]["geomean"]:.3f}' for p in c['processes']]+[f'{rows[m]["baseline_host_us"][stat]:.3f}' for stat in ['median','p10','p90']] for m in modes]),'']
            sparse={r['stage']:r['mean'] for r in rows['sparse']['trace_budget']['stages']}
            layersteps=list(dict.fromkeys(r['stage'].split('.',1)[1] for r in rows['sparse']['trace_budget']['stages'] if r['stage'].startswith('layer')))
            text+=['Per-layer sparse CUDA-active stage measurements (µs):','',table(['Layer']+layersteps,[[i]+[f'{sparse[f"layer{i}.{step}"]:.3f}' if f'layer{i}.{step}' in sparse else 'N/A' for step in layersteps] for i in range(6)]),'']
            if c['size']=='31m':
                text+=['The fused31M stage contains these separately measured CUDA activities. Their summed durations expose the short-row prepass cost and fallback cost, but do not separate the internal h/z inspection and overlapping MMA phases:','',table(['Mode','Activity role','Summed over6 layers µs'],[[m,role,f'{sum(n["mean"] for n in rows[m]["trace_budget"]["kernel_nodes"] if n["stage"].endswith("joint_gate_down_residual") and (("parallel_rows" in n["kernel"])==(role=="short-row prepass"))):.3f}'] for m in ['sparse','no_skip'] for role in ['short-row prepass','joint fallback']]),'']
            dens=[]
            for name,r in c['diagnostics']['activations'].items():
                hist=r['row_support_histogram'];total=sum(hist);short=sum(hist[:9])
                dens.append([name,f'{100*r["zeros"]/r["count"]:.2f}%',f'{100*short/total:.2f}%',f'{100*r["empty_tiles8x16"]/r["tiles8x16"]:.2f}%',f'{r["rms"]:.6f}'])
            text+=['Actual sparse-consumer gated operands, integer counts pooled over338 blocks:','',table(['Site.layer','Exact zeros','Rows ≤8 nonzeros','Empty8×16 tiles','RMS'],dens),'']
            counters=c['consumer_counters']['sites']
            if c['size']=='31m':
                text+=['Actual31M prepass decisions, pooled over338blocks. A state9 includes >8 entries or unsafe values. Both branches short for every row of an8-token group permits the fallback CTA to return early:','',table(['Layer','Rows both branches short','8-token blocks all short'],[[name,f'{100*s["both_short_row_fraction"]:.2f}%',f'{100*s["early_return_block_fraction"]:.2f}%'] for name,s in counters.items()]),'']
            else:
                text+=['Actual70M feature unions read directly from the frozen index kernel after each forward. Empty8×16tiles above are a geometric diagnostic; this consumer uses the declared GM groups and32-wide K loops. Rounded K includes MMA padding:','',table(['Site.layer','GM','Mean union K /2048','Mean rounded K /2048','Rounded K / full K'],[[name,s['gm'],f'{s["mean_union_features"]:.3f}',f'{s["mean_rounded_k"]:.3f}',f'{100*s["tensorcore_k_fraction"]:.2f}%'] for name,s in counters.items()]),'']
                numerator=sum(s['padded_tensorcore_products'] for s in counters.values());denominator=sum(s['no_skip_padded_tensorcore_products'] for s in counters.values())
                text += [f'Pooled padded tensorcore product count in these five h consumers is {numerator:,} versus {denominator:,} for their no-skip paths ({100*numerator/denominator:.3f}% retained). Integer counts are pooled before division. This is a logical executed-operand counter, not a runtime speedup;16-row MMA padding means small GM no-skip consumers are not the best dense kernels.','']
            fig,axes=plt.subplots(2,1,figsize=(11,8),layout='constrained')
            shown=[l for l in ORDER if any(cats[m].get(l,0)>0 for m in modes)];xs=list(range(len(shown)))
            for j,m in enumerate(modes):axes[0].bar([x+(j-1)*.25 for x in xs],[cats[m].get(l,0) for l in shown],width=.25,label=m,color=colors[j],hatch=['','///','...'][j])
            axes[0].set_xticks(xs,shown,rotation=45,ha='right',fontsize=8);axes[0].set_ylabel('GPU activity µs');axes[0].legend();axes[0].grid(axis='y',alpha=.2)
            by_stage={r['stage']:r for r in rows['sparse']['trace_budget']['stages']}
            for i,step in enumerate(layersteps):
                axes[1].plot(range(6),[sparse.get(f'layer{l}.{step}',float('nan')) for l in range(6)],marker=['o','s','^','D','v','P','X','<','>','*'][i%10],label=step,color=colors[i%8])
                axes[1].fill_between(range(6),[by_stage.get(f'layer{l}.{step}',{}).get('p10',float('nan')) for l in range(6)],[by_stage.get(f'layer{l}.{step}',{}).get('p90',float('nan')) for l in range(6)],color=colors[i%8],alpha=.12)
            axes[1].set_xlabel('Transformer layer');axes[1].set_ylabel('Sparse stage GPU activity µs');axes[1].set_ylim(bottom=0);axes[1].legend(ncol=4,fontsize=7,loc='upper center',bbox_to_anchor=(.5,-.24));axes[1].grid(alpha=.2)
            fig.suptitle(key.replace('0p1','0.1')+' — category comparison and sparse per-layer map');pdf.savefig(fig,bbox_inches='tight',pad_inches=.18);plt.close(fig)
    text+=['## Numerical and interpretation limits','',
      f'All12 fresh processes pass the unchanged bounds: {data["logit_comparisons"]:,} complete-output comparisons. Maximum absolute logit error across the cohort is {max(p["max_logit_absolute_error"] for c in data["conditions"].values() for p in c["processes"]):.6g}; exact equality is reported only when observed. Loss values and per-block numerical evidence remain in results/decomposition.json and quality.json.','',
      'Sparse versus native includes all frozen component optimizations (paired norms, attention/layout and head), while sparse versus no-skip isolates the net skipping effect in each existing implementation.70M uses sparse h only in five layers; z remains dense. A no-skip control that retains conversion is not the best possible dense kernel. Changing the gate structure or porting the14M corner-specialized consumer would require a separate experiment. Panels share the complete-forward horizontal scale; per-condition and per-layer detail panels use their own vertical scales. Single-seed, full-sequence batch1 results do not cover autoregressive decode, other sequence lengths, batching, training, or other GPUs. Raw CUPTI profiling and events have their own overhead and boundary definitions; no phase is assigned a fraction of uninstrumented wall time by rescaling.','',
      'Source scripts:20_analyze.py and21_report.py. Caption/legend: colors/patterns identify forward stages; bars sum actual GPU activities, lines show sparse per-layer means; bands show p10–p90 across12 profiled replays (a descriptive spread, not a confidence interval). Zero-valued layout stages are views. All figures are PDF only.','',
      '[Publication PDF](../figures/01-larger-forward-decomposition.pdf) · [All stage timings](../results/forward-stages.csv) · [Every CUDA activity](../results/cuda-kernel-nodes.csv) · [Verified result](../results/decomposition.json)']
    (dest/'01-larger-forward-decomposition.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    (dest/'INDEX.md').write_text('# Observations\n\n[01 — complete larger-model forward decomposition](01-larger-forward-decomposition.md): native/sparse/no-skip host latency, actual per-stage and per-layer GPU activities, instrumentation overhead, full numerical qualification and operand diagnostics.\n',encoding='utf-8')
    write(ROOT/'results/report-provenance.json',{'scripts':[record(ROOT/'20_analyze.py'),record(ROOT/'21_report.py')],'source':record(ROOT/'results/decomposition.json'),'14m_context':record(prior_path,ROOT.parents[1]),'outputs':[record(p) for p in sorted(dest.glob('*'))]+[record(p) for p in sorted(figures.glob('*.pdf'))]+[record(ROOT/'results'/name) for name in ['forward-stages.csv','cuda-kernel-nodes.csv','forward-categories.csv']]})
if __name__=='__main__':main()

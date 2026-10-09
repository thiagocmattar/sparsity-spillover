"""Qualified process reduction; keep diagnostic and practical claims distinct."""
import math
from common import RUN, config, sources, read, record, write


def main():
    cfg=config();verification=read(RUN/'results/verification.json')
    if verification['status']!='verified':raise ValueError('Verified final cohort required')
    rows=[];provenance=[]
    for source in sources()['checkpoints']:
        paths=[RUN/'artifacts/endpoints'/f'{source["id"]}-r{i}'/'result.json' for i in range(1,cfg['process_replicates']+1)]
        results=[read(p) for p in paths];provenance.extend(record(p) for p in paths)
        modes=set.intersection(*(set(r['timings']['full_forward']) for r in results))
        for mode in sorted(modes):
            row={'model':source['id'],'head_kappa':source['kappa'],'canonical_loss':source['production_loss'],
                 'mode':mode,'processes':len(results),'measurements':{}}
            for boundary in ('projection_only','gate_cast_projection','full_forward'):
                row['measurements'][boundary]={}
                for clock in ('host_ms','cuda_ms'):
                    values=[r['timings'][boundary][mode][clock]['geomean'] for r in results]
                    native=[r['timings'][boundary]['native'][clock]['geomean'] for r in results]
                    row['measurements'][boundary][clock]={'geomean_ms':math.exp(sum(map(math.log,values))/len(values)),
                       'process_min_ms':min(values),'process_max_ms':max(values),
                       'latency_change_vs_same_checkpoint_native_percent':100*(math.exp(sum(math.log(a/b) for a,b in zip(values,native))/len(values))-1)}
            rows.append(row)
    write(RUN/'results/comparison.json',{'status':'verified','rows':rows,'sources':provenance,
          'profiler_status':verification['profiler_status'],
          'scope':'frozen-checkpoint head implementation; full model retains native dense body; no practical quality-latency frontier gain established'})
    lines=['# Vocabulary kernel efficiency: matched final measurements','',
           'This diagnostic keeps Run068 weights, gates and quality unchanged. Native means native PyTorch/cuBLAS',
           'on the **same gated checkpoint**. It is a kernel-equivalence control. The original ungated model and',
           'existing optimized 14M frontier remain separate practical references; the strongest body pipeline is',
           'not integrated here. A kernel improvement alone does not establish a useful quality–latency gain.','',
           'All accepted modes qualify over all338 validation blocks in three fresh processes on the same RTX5090.',
           'Times below are geometric means of equal-size process geometric means. Independent host boundaries',
           'overlap and must not be summed. Positive full-forward change means slower than the matched native control.','',
           '| Head κ | Canonical loss | Mode | Projection µs | Gate/cast/projection µs | Full ms | Full change vs matched native |',
           '|---:|---:|---|---:|---:|---:|---:|']
    for r in rows:
        m=r['measurements'];p=m['projection_only']['host_ms'];h=m['gate_cast_projection']['host_ms'];f=m['full_forward']['host_ms']
        lines.append(f'| {r["head_kappa"]} | {r["canonical_loss"]:.6f} | {r["mode"]} | {p["geomean_ms"]*1000:.3f} | {h["geomean_ms"]*1000:.3f} | {f["geomean_ms"]:.6f} | {f["latency_change_vs_same_checkpoint_native_percent"]:+.2f}% |')
    lines.extend(['',f'Profiler status: **{verification["profiler_status"]}**. Counter denial or incomplete collection leaves',
                  'the hardware bottleneck unresolved; it must not be replaced by a confident source-level explanation.','',
                  'Exploratory screens retain all variants and failures. Only modes qualified in every final process',
                  'appear above. Process ranges in comparison.json are descriptive ranges, not confidence intervals.',
                  'One GPU, two checkpoints, one training seed and one uncached full-sequence workload limit generalization.','',
                  'Sources: [verification](verification.json), [process reduction](comparison.json),',
                  '[raw endpoints](../artifacts/endpoints/), [all exploratory candidates](../artifacts/screen/),',
                  '[profiler reports](../artifacts/profiles/). Generator: [11_reduce.py](../11_reduce.py).'])
    (RUN/'results/REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n')


if __name__=='__main__':main()

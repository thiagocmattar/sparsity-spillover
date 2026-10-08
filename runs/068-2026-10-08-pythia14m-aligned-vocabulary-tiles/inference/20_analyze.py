"""Reduce only the complete verified RTX5090 cohort; retain paired ratios and caveats."""
import argparse
import importlib.util
import json
import math
from statistics import mean
from io_utils import RUN, read, write, record


def geomean(values):
    return math.exp(mean(math.log(x) for x in values))


def paired_rows(path, cfg):
    rows=[json.loads(line) for line in path.read_text().splitlines()]
    indexed={(r['input'],r['pass'],r['mode']):r for r in rows}
    expected={(i,p,m) for i in range(cfg['timing_inputs']) for p in range(cfg['timing_passes']) for m in cfg['modes']}
    if set(indexed)!=expected or len(indexed)!=len(rows):raise ValueError('Unpaired or duplicate timing rows')
    if any(not math.isfinite(r[t]) or r[t]<=0 for r in rows for t in ('host_ms','cuda_ms')):raise ValueError('Invalid timing values')
    return indexed


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--tag',required=True);args=parser.parse_args()
    cfg=read(RUN/'config.json');inputs=read(RUN/'provenance/trained-inputs.json')
    if read(RUN/f'transfer/receipt-{args.tag}.json')['status']!='passed':raise ValueError('Verified local transfer required')
    spec=importlib.util.spec_from_file_location('run068_head_verifier',RUN/'05_verify.py')
    verifier=importlib.util.module_from_spec(spec);spec.loader.exec_module(verifier)
    comparisons=[];sources=[]
    filenames={'full_forward':'full-forward','gate_cast_projection':'gate-cast-projection','projection_only':'projection-only'}
    for model in inputs['checkpoints']:
        members=[];counter_sets=[]
        for replicate in range(1,cfg['process_replicates']+1):
            path=RUN/'artifacts/attempts'/f'full-{model["id"]}-{replicate}-{args.tag}'
            result=verifier.verify_result(path,cfg,model['id'],replicate);sources.append(record(path/'result.json'))
            counter_sets.append(read(path/'head-diagnostics.json')['counts'])
            timings={}
            for boundary,filename in filenames.items():
                indexed=paired_rows(path/f'samples-{filename}.jsonl',cfg)
                timings[boundary]={}
                for timer in ('host_ms','cuda_ms'):
                    keys=[(i,p) for i in range(cfg['timing_inputs']) for p in range(cfg['timing_passes'])]
                    timings[boundary][timer]={
                        'geomean_ms':{mode:geomean([indexed[i,p,mode][timer] for i,p in keys]) for mode in cfg['modes']},
                        'dense_over_sparse':geomean([indexed[i,p,'dense'][timer]/indexed[i,p,'custom_sparse'][timer] for i,p in keys]),
                        'no_skip_over_sparse':geomean([indexed[i,p,'custom_no_skip'][timer]/indexed[i,p,'custom_sparse'][timer] for i,p in keys])}
            members.append({'replicate':replicate,'timings':timings,'loss':result['loss'],
                            'max_logit_error':max(row['max_abs'] for mode in cfg['modes'] for row in read(path/'quality.json')['rows'][mode])})
        if any(c!=counter_sets[0] for c in counter_sets):raise ValueError('Native activation counts differ across processes')
        aggregate={}
        for boundary in filenames:
            aggregate[boundary]={}
            for timer in ('host_ms','cuda_ms'):
                entries=[member['timings'][boundary][timer] for member in members]
                aggregate[boundary][timer]={
                    'geomean_ms':{mode:geomean([r['geomean_ms'][mode] for r in entries]) for mode in cfg['modes']},
                    'dense_over_sparse':geomean([r['dense_over_sparse'] for r in entries]),
                    'no_skip_over_sparse':geomean([r['no_skip_over_sparse'] for r in entries]),
                    'process_dense_over_sparse':[r['dense_over_sparse'] for r in entries]}
        comparisons.append({'id':model['id'],'kappa':model['kappa'],'pressure_sites':model['pressure_sites'],
                            'timings':aggregate,'processes':members,'head_counts':counter_sets[0]})
    write(RUN/'results/comparison.json',{'rows':comparisons,'sources':sources,'coverage':cfg['validation'],
          'primary_timer':'synchronized host; CUDA events use separate replays',
          'precision':'FP32 checkpoint parameters, native FP32 head gate, FP16 projection',
          'interpretation':'16x16 gate emptiness equals 16x16 consumer skip fraction; source-level skipped loads are not measured DRAM traffic'})
    lines=['# Sparse vocabulary head - verified RTX5090 comparison','',
           'All values are milliseconds. Three fresh processes per model; 64 fixed inputs x 7 paired passes per process.',
           'The gate, weights and full-vocabulary output contract remain matched. This is B1/T2048 inference.','',
           '| Head kappa | Pressure history | Dense head path | No-skip head path | Sparse head path | Head reduction vs dense | Dense forward | Sparse forward | Forward reduction |',
           '|---:|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in comparisons:
        head=row['timings']['gate_cast_projection']['host_ms'];full=row['timings']['full_forward']['host_ms']
        h=head['geomean_ms'];f=full['geomean_ms'];pressure='Ph+head' if len(row['pressure_sites'])==2 else 'Ph'
        lines.append(f'|{row["kappa"]:g}|{pressure}|{h["dense"]:.6f}|{h["custom_no_skip"]:.6f}|{h["custom_sparse"]:.6f}|{100*(1-1/head["dense_over_sparse"]):+.2f}%|{f["dense"]:.6f}|{f["custom_sparse"]:.6f}|{100*(1-1/full["dense_over_sparse"]):+.2f}%|')
    lines += ['', 'The no-skip comparison isolates skipping within the custom package; the dense comparison includes layout and preparation differences.',
              'Projection-only timings, individual process ratios, losses and exact error maxima are retained in comparison.json.',
              'One model size, training seed, GPU and workload do not establish decoding or scale-independent performance.']
    (RUN/'results/REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n')
    print({'status':'reduced','models':len(comparisons)},flush=True)


if __name__=='__main__':main()

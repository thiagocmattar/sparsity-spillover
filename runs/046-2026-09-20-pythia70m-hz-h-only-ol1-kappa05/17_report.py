"""Reduce only the verified training endpoint and their paired K050 probes."""
import hashlib
import json
import math
from pathlib import Path
import statistics

RUN=Path(__file__).resolve().parent


def read(path):return json.loads(path.read_text())


def geomean(values):
    assert values and all(math.isfinite(x) and x>0 for x in values)
    return math.exp(statistics.mean(map(math.log,values)))


def main():
    training=read(RUN/'artifacts/verification.json')
    latency=read(RUN/'latency/artifacts/verification.json')
    assert training['status']=='verified' and training['condition_count']==1
    assert latency['all_required_artifacts_verified']
    assert len(latency['scientific_processes'])==3
    sources=[RUN/'artifacts/verification.json',RUN/'latency/artifacts/verification.json']
    rows=[]
    for index,t in enumerate(sorted(training['conditions'],key=lambda x:x['condition']['gate_threshold'])):
        ratios=[];native=[];candidate=[];qualified=[];losses=[];max_errors=[];max_l2=[];loss_deltas=[]
        for replicate in range(1,4):
            folder=RUN/'latency/artifacts/attempts'/f'scientific-c{index:02d}-r{replicate}-001'
            result=read(folder/'result.json');timing=read(folder/'timing.json')
            quality=read(folder/'quality.json')
            sources += [folder/'result.json',folder/'timing.json',folder/'quality.json']
            assert result['status']=='complete' and result['validation_blocks']==338
            assert result['checkpoint']['final_checkpoint_content_sha256']==t['checkpoint_content_sha256']
            qualified.append(result['qualified']);losses.append(result['loss'])
            max_errors.append(max(g['max_abs'] for g in quality['gates']['candidate_graph']))
            max_l2.append(max(g['relative_l2'] for g in quality['gates']['candidate_graph']))
            loss_deltas.append(abs(quality['loss_delta']['candidate_graph']))
            pairs={}
            for sample in timing['samples']:
                key=(sample['repeat'],sample['input_index'])
                pair=pairs.setdefault(key,{})
                assert sample['mode'] not in pair
                pair[sample['mode']]=sample['host_ms']
            assert len(pairs)==64*7 and len(timing['indices'])==64
            for pair in pairs.values():
                assert set(pair)=={'native_graph','candidate_graph'}
                native.append(pair['native_graph']);candidate.append(pair['candidate_graph'])
                ratios.append(pair['native_graph']/pair['candidate_graph'])
        rows.append({'kappa':t['condition']['gate_threshold'],
            'training_validation_loss':t['final_validation_loss'],
            'h_exact_zero_fraction':t['selected_site_exact_zero_fractions']['h'],
            'z_exact_zero_fraction':t['selected_site_exact_zero_fractions']['z'],
            'R_model':t['R_model'],'R_model_max':t['R_model_max'],
            'paired_samples':len(ratios),'qualified':all(qualified),
            'maximum_k050_logit_absolute_error':max(max_errors),
            'maximum_k050_logit_relative_l2':max(max_l2),
            'maximum_k050_loss_delta':max(loss_deltas),
            'native_geomean_host_ms':geomean(native),
            'k050_geomean_host_ms':geomean(candidate),
            'native_relative_paired_geomean_speedup':geomean(ratios),
            'bf16_qualification_losses':losses})
    provenance=[]
    for path in sources:
        provenance.append({'path':path.relative_to(RUN).as_posix(),
            'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    report={'conditions':rows,'sources':provenance,
        'latency_unit':'ms per B=1,T=2048 full-vocabulary uncached forward',
        'aggregation':'geometric mean of all 1344 balanced paired host-time ratios per condition',
        'post_hoc_clipping':'none'}
    (RUN/'artifacts/summary.json').write_text(json.dumps(report,indent=2)+'\n')
    lines=['# 001 — One h/z threshold endpoint and final K050 latency','',
        '**Question.** How do the approved h/z threshold of 0.5 behave with h-only OL1, lambda=1?', '',
        '**Method and coverage.** One random-initialized Pythia70M model matches Run043 initialization, '
        'seed, MiniPile order and 712-step budget. Gates act only at h and z; pressure acts '
        'only at post-gate h. No post-hoc clipping. Training validation covers all 338 complete '
        'blocks from 500 documents (1,444 tail tokens excluded). The frozen Run035 '
        'k050-70m-v2 kernel, with the HZ allowlist bridge, uses RTX5090, BF16, '
        'B=1, T=2048, full logits, three fresh processes and 64 inputs × 7 passes each. '
        'Every process receives complete numerical qualification against native eager logits. '
        'The speedup denominator is native CUDA-graph inference of the same checkpoint.', '',
        '**Result.** Exact-zero fractions pool integer counts. Latencies and paired speedups '
        'below are geometric means over 1,344 pairs per condition.', '',
        '| kappa | Training val loss | h zero % | z zero % | R_model % | Native ms | K050 ms | Speedup | Qualified |',
        '|---:|---:|---:|---:|---:|---:|---:|---:|:---:|']
    for r in rows:
        lines.append(f"| {r['kappa']:g} | {r['training_validation_loss']:.6f} | "
            f"{100*r['h_exact_zero_fraction']:.3f} | {100*r['z_exact_zero_fraction']:.3f} | "
            f"{100*r['R_model']:.3f} | {r['native_geomean_host_ms']:.6f} | {r['k050_geomean_host_ms']:.6f} | "
            f"{r['native_relative_paired_geomean_speedup']:.4f}x | {r['qualified']} |")
    lines += ['', f"**Numerical qualification.** {sum(r['qualified'] for r in latency['scientific_processes'])}/3 "
        'processes passed complete 338-block qualification. Maximum kernel logit absolute '
        f"error: {max(r['maximum_k050_logit_absolute_error'] for r in rows):.9g}; relative L2: "
        f"{max(r['maximum_k050_logit_relative_l2'] for r in rows):.9g}; absolute pooled loss "
        f"difference: {max(r['maximum_k050_loss_delta'] for r in rows):.9g}. "
        'The fixed logit bound is abs(error) <= 0.25 + 0.02 * abs(reference), '
        'with relative L2 <= 0.02 and absolute pooled loss difference <= 0.001.', '',
        '**Runtime result.** ' + (
            'The numerically qualified specialized kernel is '
            + ('faster than' if rows[0]['native_relative_paired_geomean_speedup']>1 else
               'slower than' if rows[0]['native_relative_paired_geomean_speedup']<1 else 'as fast as')
            + ' native graph inference for this condition on this workload.'
            if rows[0]['qualified'] else
            'The specialized kernel did not qualify; its timing is not evidence of valid acceleration.'), '',
        '**Interpretation limits.** One seed, one model size and one data pass. '
        'This joint intervention does not isolate the effects of either gate or pressure. '
        'Logical-product opportunity is distinct from measured speedup. BF16 qualification '
        'losses are retained separately from training validation. Timing of an unqualified '
        'condition, if any, is not evidence of a valid accelerated implementation. '
        'No manuscript or research finding is promoted.', '',
        '**Source and provenance.** `../17_report.py`, `../artifacts/summary.json`, '
        '`../artifacts/verification.json`, `../latency/artifacts/verification.json`, '
        'and the raw artifacts listed with SHA-256 identities in the summary. '
        'No figure was required; the table above is the complete single-condition result.', '']
    observations=RUN/'observations';observations.mkdir(exist_ok=True)
    (observations/'001-final-results.md').write_text('\n'.join(lines),encoding='utf-8')
    (observations/'INDEX.md').write_text('# Run046 observations\n\n'
        '- [001: Final training and K050 latency](001-final-results.md).\n')
    print(json.dumps(rows,indent=2))


if __name__=='__main__':main()

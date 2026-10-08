"""Reduce only verified terminal training into loss and aligned-head diagnostics."""
import hashlib
import json
import math
from run_config import RUN_DIR,load_config,condition_specs,write_json
from head_site import HEAD_NAME


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def identity(path):
    with path.open('rb') as handle:digest=hashlib.file_digest(handle,'sha256').hexdigest()
    return {'path':path.relative_to(RUN_DIR).as_posix(),'bytes':path.stat().st_size,'sha256':digest}


def main():
    config=load_config();verification=read(RUN_DIR/'artifacts/verification.json')
    if verification['status']!='verified' or verification['condition_count']!=2:
        raise ValueError('Both verified terminal training conditions required')
    rows=[];sources=[]
    for condition in condition_specs(config):
        matches=[p for p in (RUN_DIR/'artifacts/attempts').iterdir() if p.is_dir()
                 and read(p/'metrics.json')['condition']['id']==condition['id']]
        if len(matches)!=1:raise ValueError('Exactly one terminal attempt per condition required')
        attempt=matches[0];metrics=read(attempt/'metrics.json')
        stats=read(attempt/'diagnostics/activation_statistics.json')
        head=next(r for r in stats['rows'] if r['name']==HEAD_NAME)
        pre=next(r for r in stats['rows'] if r['name']=='head_in_pre.final')
        tiles=next(r for r in stats['occupancy_rows'] if r['name']==HEAD_NAME)
        corners=next(r for r in stats['corner_rows'] if r['name']==HEAD_NAME)
        energy=next(r for r in stats['sign_energy_rows'] if r['name']==HEAD_NAME)
        if head['total']!=88604672 or corners['tiles']!=346112 or corners['certificate_mismatches']:
            raise ValueError('Incomplete aligned head count/certificate')
        loss=metrics['validation']['step_000712']['loss']
        rows.append({'id':condition['id'],'head_kappa':condition['head_gate_threshold'],
            'block_kappa':0.,'head_gate':condition['head_gate'],'block_gate':condition['block_gate'],
            'pressure_sites':condition['pressure_sites'],'loss':loss,'perplexity':math.exp(loss),
            'head_elements':head['total'],'head_exact_zero_count':head['exact_zero_count'],
            'head_exact_zero_fraction':head['exact_zero_count']/head['total'],
            'head_pre_rms':math.sqrt(pre['square_sum']/pre['finite']),
            'head_post_rms':math.sqrt(head['square_sum']/head['finite']),
            'head_removed_energy_fraction':(energy['rejected_square_sum']/energy['pre_square_sum'] if energy['pre_square_sum'] else None),
            'head_tiles':tiles['tiles'],'head_corner':corners,
            'validation':metrics['validation']['step_000712'],'source':attempt.relative_to(RUN_DIR).as_posix()})
        sources += [identity(attempt/n) for n in ('metrics.json','diagnostics/activation_statistics.json','diagnostics/logical_products.json')]
    delta=rows[1]['loss']-rows[0]['loss']
    write_json(RUN_DIR/'results/training-comparison.json',{'rows':rows,'sources':sources,
        'higher_threshold_loss_delta':delta,'scope':'native dense B32/T2048 training evaluation; not latency or causal tile-alignment effect'})
    lines=['# Aligned vocabulary-head training comparison','',
        'Both models: h/z8x16 kappa0; head16x16; Ph+head with equal seven-tensor OL1 pressure.',
        'Full500-document/338-block validation;692224 input tokens,1444 excluded tail.','',
        '| Head kappa | Loss | Perplexity | Head exact zeros | Empty16x16 tiles | Head RMS | Removed energy |',
        '|---:|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        tile=row['head_tiles']['16x16']
        removed='undefined' if row['head_removed_energy_fraction'] is None else f"{100*row['head_removed_energy_fraction']:.3f}%"
        lines.append(f"|{row['head_kappa']:g}|{row['loss']:.6f}|{row['perplexity']:.3f}|{100*row['head_exact_zero_fraction']:.3f}%|{100*tile['empty']/tile['total']:.3f}%|{row['head_post_rms']:.6f}|{removed}|")
    lines += ['',f'Paired head-kappa.5 minus.1 loss delta: {delta:+.6f}.',
        'These are B32 native activation counts. Actual B1 inference bypass and latency are measured separately in inference/results/REPORT.md.',
        'The two models isolate head threshold at fixed alignment. Historical Run066 comparisons do not isolate alignment alone.',
        'One seed, architecture and budget; no acceptable-quality cutoff or broader speedup is inferred.']
    (RUN_DIR/'results/TRAINING.md').write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n')
    print({'status':'reduced','conditions':len(rows)},flush=True)


if __name__=='__main__':main()

"""Recompute global OL1 geometry from all retained training boundaries."""
import importlib.util
import json
import math
import numpy as np
from evidence import ROOT, SOURCES, identity, read, run, verify, write

AUDITOR = ROOT/'analyses/021-2026-09-10-training-results-figures/02_ol1_geometry.py'
spec = importlib.util.spec_from_file_location('retained_geometry', AUDITOR)
geometry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(geometry)


def quantiles(values):
    return dict(zip(('p25','median','p75'), np.quantile(values, [.25,.5,.75]).tolist()))


def reduce_condition(attempt, descriptor, expected_sites, historical_h_only=False):
    manifest = read(attempt/'manifest.json')
    assert manifest['status']=='completed' and manifest['completed_steps']==712
    pressure = manifest['activation_pressure']
    assert pressure['method']=='orthogonal_l1' and pressure['weight']==pressure['step_budget']==1
    assert pressure['eps']==1e-12
    if not historical_h_only:
        assert pressure['sites']==expected_sites
    path=attempt/'events.jsonl'
    identity(path)
    rows=[json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
    rows=[r for r in rows if r.get('event')=='train']
    assert [r['step'] for r in rows]==list(range(1,713))
    assert len({r['condition_id'] for r in rows})==1
    trace={key:[] for key in ('rho_task_percent','removed_pressure_percent','pre_projection_norm_ratio',
        'pre_projection_pressure_norm','pre_cap_ratio','raw_gradient_conflict','adaptive_conflict','cap_active')}
    for row in rows:
        geometry.audit_step(row, pressure)
        assert row['gradient_conflict']==(row['task_pressure_gradient_dot']<0)
        if 'pressure_capture_names' in row:
            expected=[f'h.layer_{i}' for i in range(6)] + (['head_in.final'] if 'head_in' in expected_sites else [])
            assert row['pressure_capture_names']==sorted(expected)
        elif not historical_h_only:
            assert row['pressure_capture_tensor_count']==6*len(expected_sites)
        u,w=row['task_direction_norm'],row['pressure_direction_norm_raw']
        alpha=geometry.opposing_component_ratio(row)
        removed=alpha*u/w
        assert -1e-14<=removed<=1+1e-12
        trace['rho_task_percent'].append(100*alpha)
        trace['removed_pressure_percent'].append(100*removed)
        trace['pre_projection_norm_ratio'].append(w/(u+pressure['eps']))
        trace['pre_projection_pressure_norm'].append(w)
        trace['pre_cap_ratio'].append(row['pressure_to_task_ratio_raw']/pressure['step_budget'])
        trace['raw_gradient_conflict'].append(row['gradient_conflict'])
        trace['adaptive_conflict'].append(row['projection_applied'])
        trace['cap_active'].append(row['trust_scale']<1)
    summary={k:quantiles(v) for k,v in trace.items() if not isinstance(v[0],bool)}
    for key in ('raw_gradient_conflict','adaptive_conflict','cap_active'):
        summary[key]={'count':sum(trace[key]),'total':len(rows),'percent':100*sum(trace[key])/len(rows)}
    summary['eligible_parameter_tensors']=sorted({r['eligible_parameter_tensors'] for r in rows})
    summary['skipped_parameter_tensors']=sorted({r['skipped_parameter_tensors'] for r in rows})
    return dict(descriptor, source_attempt=attempt.relative_to(ROOT).as_posix(),
        pressure_sites=expected_sites, declared_pressure_sites=pressure['sites'],
        historical_h_only_audit=historical_h_only, steps=712, zero_lr_boundaries=sum(r['learning_rate']==0 for r in rows),
        summary=summary, trace=trace, initial_parameter_sha256=manifest['initial_parameter_sha256'],
        training_schedule_hash=manifest['training_schedule_hash'], pressure=pressure)


def main():
    identity(AUDITOR)
    conditions=[]
    for prefix in ('060','066','068'):
        r=run(prefix)
        for attempt in sorted((r/'artifacts/attempts').iterdir()):
            if not attempt.is_dir():continue
            manifest=read(attempt/'manifest.json')
            c=manifest['condition']
            block=c.get('block_gate',{}).get('kappa',c.get('gate_threshold'))
            head=c.get('head_gate_threshold',c.get('gate_threshold')) if prefix!='060' else None
            joint='head_in' in c['pressure_sites']
            family='T2 tile / Ph' if prefix=='060' else 'Head8 / Ph+head' if prefix=='066' and joint else 'Head8 / Ph' if prefix=='066' else 'Head16 / Ph+head'
            label=(f'No head | hz={block:g}' if prefix=='060' else
                   f'H8 {"Ph+head" if joint else "Ph"} | hz=head={head:g}' if prefix=='066' else
                   f'H16 Ph+head | hz=0, head={head:g}')
            conditions.append(reduce_condition(attempt,dict(id=c['id'],family=family,label=label,
                block_kappa=block,head_kappa=head,kappa=head if head is not None else block,
                head_tile=0 if prefix=='060' else 8 if prefix=='066' else 16,
                pressure_recipe='Ph+head' if joint else 'Ph',role='head comparison',run=prefix),c['pressure_sites']))
    assert len(conditions)==8
    assert len({r['initial_parameter_sha256'] for r in conditions})==1
    assert len({r['training_schedule_hash'] for r in conditions})==1
    appendix=read('analyses/024-2026-09-17-h-only-kernel-latency/data/ol1-appendix.json')
    for path,digest in appendix['source_sha256'].items():
        if path.endswith(('events.jsonl','manifest.json')) or path==AUDITOR.relative_to(ROOT).as_posix():
            verify({'path':path,'sha256':digest})
    historical=[]
    for c in appendix['geometry']['conditions']:
        if c['model']!='14M':continue
        descriptor=dict(id=f"paper-T{c['scope']}-{c['pressure']}-{c['kappa']}",
            family=f"T{c['scope']} / P{c['pressure']}",label=f"T{c['scope']}/P{c['pressure']} k={c['kappa']:g}",
            kappa=c['kappa'],block_kappa=c['kappa'],head_kappa=None,head_tile=0,
            pressure_recipe='Ph' if c['pressure']=='h' else 'Pall',role='manuscript context')
        row=reduce_condition(ROOT/c['source_attempt'],descriptor,c['realized_pressure_sites'],c['historical_h_only_audit'])
        assert math.isclose(row['summary']['rho_task_percent']['median'],100*c['rho_opp']['median'],rel_tol=1e-12)
        assert math.isclose(row['summary']['pre_cap_ratio']['median'],c['median_pre_cap_ratio'],rel_tol=1e-12)
        historical.append(row)
    assert len(historical)==20
    pairs=[]
    for kappa in (0,.1):
        h=next(r for r in conditions if r['family']=='Head8 / Ph' and r['kappa']==kappa)
        joint=next(r for r in conditions if r['family']=='Head8 / Ph+head' and r['kappa']==kappa)
        pairs.append(dict(kappa=kappa,ph=h['id'],ph_head=joint['id'],
            delta_median_rho_task_percent=joint['summary']['rho_task_percent']['median']-h['summary']['rho_task_percent']['median'],
            delta_adaptive_conflict_pp=joint['summary']['adaptive_conflict']['percent']-h['summary']['adaptive_conflict']['percent'],
            delta_raw_conflict_pp=joint['summary']['raw_gradient_conflict']['percent']-h['summary']['raw_gradient_conflict']['percent']))
    write('data/geometry.json',dict(conditions=conditions,manuscript_context=historical,matched_pressure_pairs=pairs,
        boundary_records=712*(len(conditions)+len(historical)),definitions={
            'u':'Task Adam direction over parameters eligible for both task and pressure; no group LR or weight decay.',
            'w':'Unweighted pressure gradient preconditioned with the same task second moment; before projection.',
            'rho_task_percent':'100*max(0,-dot(u,w))/(||u||^2+eps): removed component norm relative to task direction (manuscript definition).',
            'removed_pressure_percent':'100*||w-w_safe||/||w||: norm of removed component relative to pressure, not energy or coordinate count.',
            'pre_projection_norm_ratio':'||w||/(||u||+eps), measured before conflict projection.',
            'pre_cap_ratio':'lambda*||w_safe||/(||u||+eps)/b, after projection and before norm cap (manuscript r/b).',
            'raw_gradient_conflict':'Count(g_task dot g_pressure < 0)/712.',
            'adaptive_conflict':'Count(dot(u,w)<0 and ||u||^2>eps)/712; equals projection_applied.',
            'cap_active':'Count(trust_scale<1)/712, including the zero-LR first boundary.'},
        limits='Global pooled geometry cannot isolate the head contribution. Ph uses 6 tensors, Ph+head 7; weighting and eligible parameter pool both change. One seed.'))
    identity(__file__)
    write('provenance/geometry-sources.json',list(SOURCES.values()))
    print({'status':'geometry-collected','head_comparison_conditions':8,'manuscript_context_conditions':20,'boundaries':19936},flush=True)


if __name__=='__main__':
    main()

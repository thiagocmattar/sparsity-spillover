"""Check the revision's quoted numerical comparisons against unrounded data."""
from pathlib import Path
import hashlib
import json
import math

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]


def read(p):return json.loads(p.read_text(encoding='utf-8'))


def main():
    old=read(ROOT/'analyses/018-2026-09-08-results-materials/figure_data.json')
    trained={r['id']:r for r in old['trained']}
    cross=read(HERE/'data/cross-size-audit.json')
    logs=read(HERE/'data/log-audit.json')
    runtime=read(ROOT/'analyses/019-2026-09-09-manuscript-rewrite-audit/runtime-audit.json')
    checks=[]
    def quoted(name,actual,text,digits):
        assert f'{actual:.{digits}f}'==text,(name,actual,text)
        checks.append(dict(claim=name,unrounded=actual,display=text,decimal_places=digits))
    for scale,loss,sp,penalty,delta,ds in [
        ('14M','5.8294','27.483','0.6208','-0.2086','14.769'),
        ('70M','5.2159','40.602','1.1162','-0.1736','5.006'),
        ('410M','5.1207','80.616','0.5732','-0.0703','9.024')]:
        a0=trained[f'{scale}:A0:None'];a4=trained[f'{scale}:A4-OL1:0.5'];a7=trained[f'{scale}:A7-OL1:0.5']
        for n,x,t,d in [('A7 loss',a7['loss'],loss,4),('A7 S percent',100*a7['R_model'],sp,3),
                        ('A7 penalty',a7['loss']-a0['loss'],penalty,4),
                        ('A7 minus A4 loss',a7['loss']-a4['loss'],delta,4),
                        ('A7 minus A4 S pp',100*(a7['R_model']-a4['R_model']),ds,3)]:quoted(scale+' '+n,x,t,d)
        low4=trained[f'{scale}:A4-OL1:0.0'];low7=trained[f'{scale}:A7-OL1:0.0']
        assert (low7['loss']<low4['loss'] and low7['R_model']>low4['R_model']) if scale=='410M' else (low4['loss']<low7['loss'] and low4['R_model']>low7['R_model'])
    for fam,loss,sp in [('A4','0.3783','2.498'),('A7','0.1265','12.096')]:
        a=trained[f'14M:{fam}:0.5'];b=trained[f'14M:{fam}-OL1:0.5']
        quoted(fam+' pressure loss increment',b['loss']-a['loss'],loss,4)
        quoted(fam+' pressure sparsity increment pp',100*(b['R_model']-a['R_model']),sp,3)
    for r,text in zip(cross['relu'],['0.0611','0.1230','0.1038']):
        quoted(r['scale']+' ReLU base cost',r['base_relu_cost'],text,4)
        p5=next(x for x in r['matched_policy'] if x['p']==.5)
        assert p5['relu_dominates']
        assert all(x['incremental_loss_difference']<0 for x in r['matched_policy'] if x['p']>0)
        assert all(x['loss_difference']>0 for x in r['matched_policy'] if x['p']<=.3)
        if r['scale']=='14M':assert r['matched_policy'][-1]['loss_difference']>0
        if r['scale']=='410M':assert all(x['loss_difference']>0 for x in r['matched_policy'] if x['p']>=.8)
    p6=cross['relu'][0]['matched_policy'][6]
    for f,t_loss,t_s,t_increment in [('gelu','6.8212','7.764','1.6126'),('relu','6.5215','7.893','1.2518')]:
        r=p6[f];quoted('14M p=.6 '+f+' loss',r['loss'],t_loss,4)
        quoted('14M p=.6 '+f+' S',100*r['S_model'],t_s,3)
        quoted('14M p=.6 '+f+' incremental loss',r['delta_loss_from_p0'],t_increment,4)
    assert len(cross['canonical_manifest_identities'])==36
    s=[r for r in logs['saturation'] if r['phase']=='all']
    assert sum(r['boundaries'] for r in s)==24208 and sum(r['cap_active'] for r in s)==7283
    quoted('OL1 pooled cap percent',100*7283/24208,'30.09',2)
    for scale,text in [('14M','0.317'),('70M','0.295'),('410M','0.850')]:
        r=next(r for r in logs['dynamics'] if r['id']==f'{scale}:A0:None')
        quoted(scale+' late A0 preclip median',r['late_preclip_norm']['median'],text,3)
        assert r['late_loss_slope_per_billion_tokens']<0
    assert all(r['late_loss_slope_per_billion_tokens']<0 for r in logs['dynamics'] if r['scale']=='410M')
    for key,text in [('k050','1.2340'),('k050-no-skip','1.1829'),('k050-attention-dense','1.2506')]:
        quoted(key+' native-relative GM',runtime['candidate_summary'][key]['geomean_native_relative'],text,4)
    quoted('sparse-path GM',runtime['sparse_path_geomean'],'1.0432',4)
    for condition,loss,latency in [('c15','5.6623','0.4790'),('c25','5.7076','0.4789')]:
        r=next(r for r in runtime['rows'] if r['condition']==condition)
        assert r['family'] in ['A4','A7'] and r['dose']==.5
        quoted(condition+' BF16 reference loss',r['BF16_validation_loss'],loss,4)
        quoted(condition+' K050 latency ms',r['candidate_latency_ms']['k050'],latency,4)
    for r in runtime['rows']:
        assert math.isclose(r['sparse_path_factor'],r['native_relative_speedups']['k050']/r['native_relative_speedups']['k050-no-skip'],abs_tol=1e-14)
        assert r['attention_dense_over_K050']>1
    fig=read(HERE/'figures/SOURCES.json')
    assert fig['cross_size_main']['evaluations']==96
    assert [len(r['ids']) for r in fig['cross_size_main']['omitted']]==[6,6,3,5,5,6]
    baseline=read(ROOT/'manuscript/draft/revision-v2/baseline.json')
    assert hashlib.sha256((ROOT/'manuscript/draft/task.md').read_bytes()).hexdigest()==baseline['task_sha256']
    out=dict(status='Verified',quoted_checks=checks,conditions=54,clipped=540,paired_14M=29,scale_pairs=15,
             cross_size_records=396,displayed_distinct_evaluations=96,OL1_boundaries=24208,
             training_dynamics_records=6408,task_bytes_preserved=True,
             interpretation='Ideal geometry is tested separately; empirical checks establish observed comparisons, not seed robustness or a causal training-budget explanation.')
    (HERE/'data/claim-checks.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(f'{len(checks)} quoted quantities plus identity, ordering, cap, omission and runtime-ratio checks pass.')


if __name__=='__main__':main()

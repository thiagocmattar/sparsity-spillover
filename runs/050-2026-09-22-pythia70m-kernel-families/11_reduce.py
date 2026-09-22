"""Reduce frozen final timings; never select a policy from validation."""
import argparse,math
import numpy as np
from support import RUN,read,write,sha

def paired_interval(trials,reference,candidate,draws=4000):
 rows=[]
 for trial in trials:
  values={(s['repeat'],s['input_index'],s['mode']):s['host_ms'] for s in trial['samples']}
  assert len(values)==len(trial['samples']),'Duplicate timing cell'
  rows.append([np.mean([math.log(values[(j,i,reference)]/values[(j,i,candidate)]) for j in range(7)]) for i in range(64)])
 x=np.asarray(rows);assert x.shape==(3,64)
 rng=np.random.default_rng(2504);stratified=[];hierarchical=[]
 for _ in range(draws):
  ix=rng.integers(0,64,(3,64));stratified.append(x[np.arange(3)[:,None],ix].mean())
  process=rng.integers(0,3,3);hierarchical.append(x[process[:,None],ix].mean())
 return {'speedup':float(np.exp(x.mean())),
         'time_reduction_fraction':float(1-np.exp(-x.mean())),
         'process_speedups':np.exp(x.mean(1)).tolist(),
         'stratified_input_block_95':np.exp(np.quantile(stratified,[.025,.975])).tolist(),
         'hierarchical_process_input_95':np.exp(np.quantile(hierarchical,[.025,.975])).tolist(),
         'method':'Average seven paired log ratios per input. Resample64 whole input blocks within each of three fixed process strata; sensitivity interval also resamples the three processes.4000 draws,seed2504.'}

def main():
 p=argparse.ArgumentParser();p.add_argument('--pipeline',default='final-001');p.add_argument('--selection',default='selection-final.json');a=p.parse_args()
 selection=read(RUN/'provenance'/a.selection);primary=selection['primary']+'_graph';results={};sources=[];timings={}
 for cid in ('c00','c24','c25'):
  runs=[];trials=[];qualities=[]
  for rep in (1,2,3):
   root=RUN/'artifacts'/f'{a.pipeline}-{cid}-r{rep}'
   r,t,q=[read(root/name) for name in ('result.json','timing.json','quality.json')]
   assert r['status']=='complete' and r['arguments']['phase']=='final'
   assert r['selection']['sha256']==sha(RUN/'provenance'/a.selection)
   assert q['blocks']==338 and q['documents']==500 and q['excluded_tail_tokens']==1444 and q['prediction_tokens']==691886
   assert t['indices']==np.random.default_rng(2504).choice(338,64,replace=False).tolist()
   assert all(v['samples']==448 for v in t['summary'].values())
   assert all(s['output_shape']==[1,2048,50304] and s['host_ms']>0 and s['cuda_ms']>0 for s in t['samples'])
   for name in ('result.json','timing.json','quality.json'):sources.append({'path':(root/name).relative_to(RUN).as_posix(),'sha256':sha(root/name)})
   runs.append(r);trials.append(t);qualities.append(q)
  modes=list(runs[0]['timing']);summary={}
  for mode in modes:
   latencies=[r['timing'][mode]['geomean_host_ms'] for r in runs]
   gates=[g for q in qualities for g in q['gates'][mode]]
   assert len(gates)==338*3
   summary[mode]={'geomean_host_ms':math.exp(math.fsum(map(math.log,latencies))/3),
                  'process_ms':latencies,'qualified':all(r['qualification'][mode] for r in runs),
                  'process_loss':[r['loss'][mode] for r in runs],'process_loss_delta':[r['loss_delta'][mode] for r in runs],
                  'failed_blocks':sum(not g['pass'] for g in gates),'max_relative_l2':max(g['relative_l2'] for g in gates),'max_absolute_logit_difference':max(g['max_abs'] for g in gates)}
  dense=[m for m in ('native_graph','native_hz_graph','dense_fused_graph','dense_policy_graph') if summary[m]['qualified']]
  best=min(dense,key=lambda m:summary[m]['geomean_host_ms'])
  contrasts={}
  for mode in (m+'_graph' for m in selection['policies']):
   if mode in summary:contrasts[mode]=paired_interval(trials,best,mode)
  if 'sparse_c_no_skip_graph' in summary:contrasts['sparse_c_no_skip_to_sparse']=paired_interval(trials,'sparse_c_no_skip_graph','sparse_c_graph')
  contrasts['gate_fusion_dense_to_native_hz']=paired_interval(trials,'native_hz_graph','dense_policy_graph')
  results[cid]={'implementations':summary,'best_qualified_dense_hz':best,'contrasts':contrasts,'native_loss':[r['loss']['native'] for r in runs]}
  timings[cid]=trials
 base=results['c00']['implementations'];support={}
 for cid in ('c24','c25'):
  result=results[cid];mode=result['implementations'][primary];contrast=result['contrasts'].get(primary)
  base_ratios=[b/x for b,x in zip(base['native_hz_graph']['process_ms'],mode['process_ms'])]
  base_route=[x/b for x,b in zip(base[primary]['process_ms'],base['native_hz_graph']['process_ms'])]
  support[cid]={'primary':primary,'qualified':mode['qualified'],
                'vs_native_Base_speedup':base['native_graph']['geomean_host_ms']/mode['geomean_host_ms'],
                'vs_efficient_Base_speedup':base['native_hz_graph']['geomean_host_ms']/mode['geomean_host_ms'],
                'process_speedups_vs_efficient_Base':base_ratios,'process_Base_route_overhead_fraction':[x-1 for x in base_route],
                'five_percent_faster_than_best_dense':bool(contrast and contrast['time_reduction_fraction']>=.05),
                'paired_interval_favors_sparse':bool(contrast and contrast['stratified_input_block_95'][0]>1),
                'hierarchical_interval_favors_sparse':bool(contrast and contrast['hierarchical_process_input_95'][0]>1),
                'beats_efficient_Base_each_process':all(x>1 for x in base_ratios),
                'Base_route_within_two_percent_each_process':all(x<=1.02 for x in base_route)}
  support[cid]['all_support_criteria']=all(support[cid][k] for k in ('qualified','five_percent_faster_than_best_dense','paired_interval_favors_sparse','beats_efficient_Base_each_process','Base_route_within_two_percent_each_process'))
 report={'status':'complete','selection_sha256':sha(RUN/'provenance'/a.selection),'selection_primary':selection['primary'],
         'coverage':{'documents':500,'blocks':338,'input_tokens':692224,'prediction_tokens':691886,'excluded_tail_tokens':1444,'timing_inputs':64,'timing_passes':7,'processes_per_checkpoint':3},
         'results':results,'support':support,'sources':sources,
         'limits':'Existing checkpoints,one training seed,one RTX5090,BF16 batch1 full2048-token forward. Training-only frozen policies. Confidence intervals do not estimate training-seed or GPU-population uncertainty. Cross-checkpoint Base ratios compare fresh processes within this retained session, not within-process paired timings.'}
 write(RUN/'results/final-summary.json',report)
 print({'support':support,'latency_ms':{cid:{m:round(v['geomean_host_ms'],6) for m,v in r['implementations'].items()} for cid,r in results.items()}})

if __name__=='__main__':main()

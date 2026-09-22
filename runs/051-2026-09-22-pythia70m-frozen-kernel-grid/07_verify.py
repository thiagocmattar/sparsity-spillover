"""Audit the completed table against raw timings, numerical gates and inventory."""
import math
from support import RUN,read,write,sha

def main():
 cfg=read(RUN/'config.json');summary=read(RUN/'results/final-summary.json');table=read(RUN/'results/complete-table.json')
 assert [r['id'] for r in table['rows']]==cfg['conditions']
 for item in read(RUN/'provenance/source-freeze.json')['files']:assert sha(RUN/item['path'])==item['sha256'],item['path']
 for item in read(RUN/'provenance/kernel-reuse.json')['files']:assert sha(RUN/item['path'])==item['sha256'],item['path']
 assert summary['selection_sha256']==sha(RUN/'provenance/selection-final.json')
 samples=0;cells=0;qualified=0;comparisons=0;failed=[]
 for row in table['rows']:
  cid=row['id'];trials=[]
  for rep in (1,2,3):
   root=RUN/'artifacts'/f'final-001-{cid}-r{rep}'
   result=read(root/'result.json');quality=read(root/'quality.json');timing=read(root/'timing.json');trials.append(timing)
   assert result['status']=='complete' and result['validation_blocks']==338
   assert quality['prediction_tokens']==691886 and quality['documents']==500 and quality['excluded_tail_tokens']==1444
   assert result['selection']['sha256']==summary['selection_sha256']
   for mode in timing['summary']:
    gates=quality['gates'][mode];assert len(gates)==338
    okay=all(g['pass'] for g in gates) and abs(quality['loss_delta'][mode])<=cfg['numerical_bounds']['validation_loss_atol']
    assert quality['pass'][mode]==okay==result['qualification'][mode]
    cells+=1;qualified+=okay;comparisons+=len(gates)
    if not okay:failed.append({'condition':cid,'replicate':rep,'mode':mode,'failed_blocks':[g['input_index'] for g in gates if not g['pass']],'loss_delta':quality['loss_delta'][mode]})
   samples+=len(timing['samples'])
  for mode,expected in summary['results'][cid]['implementations'].items():
   raw=[x['host_ms'] for t in trials for x in t['samples'] if x['mode']==mode]
   assert len(raw)==1344 and all(x>0 for x in raw)
   actual=math.exp(math.fsum(math.log(x) for x in raw)/len(raw))
   assert math.isclose(actual,expected['geomean_host_ms'],rel_tol=1e-12)
   assert math.isclose(actual,row['latency_ms'][mode],rel_tol=1e-12)
  diagnostics=RUN/'artifacts'/f'final-001-diagnostics-{cid}'
  modes=read(diagnostics/'result.json')['implementations']
  assert modes==(['dense_policy'] if cid=='c00' else ['dense_policy','sparse_c'])
  for mode in modes:
   for kind in ('diagnostics','structure','work','compiler','profile-summary'):assert (diagnostics/f'{kind}-{mode}.json').is_file()
   assert read(diagnostics/f'diagnostics-{mode}.json')['coverage']['blocks']==338
  assert all(abs(row['pooled_BF16_activations'][site]['zero_fraction']-row['pooled_BF16_activations'][site]['exact_zero']/row['pooled_BF16_activations'][site]['total'])<1e-15 for site in ('h','z'))
 retention=read(RUN/'results/retention-audit.json');assert not retention['missing'] and not retention['different']
 report={'status':'verified','checkpoint_count':len(table['rows']),'fresh_benchmark_processes':33,'graph_backend_cells':cells,'qualified_graph_backend_cells':qualified,'block_comparisons':comparisons,'raw_timing_observations':samples,'failed_cells':failed,'full_diagnostic_model_passes':21,'retained_artifact_files':retention['remote_files'],'retained_artifact_bytes':retention['remote_bytes'],'source_sha256':sha(RUN/'07_verify.py'),'table_sha256':sha(RUN/'results/complete-table.json')}
 write(RUN/'results/verification.json',report);print(report)

if __name__=='__main__':main()

"""Create the complete qualified table from this session's measured artifacts."""
import math
from support import RUN,read,write,sha

def main():
 summary=read(RUN/'results/final-summary.json')
 catalog=read(RUN/'provenance/inputs.json')['checkpoints']
 base=summary['results']['c00']['implementations']
 base_pt=base['native_graph']['geomean_host_ms']
 base_dense=base['native_hz_graph']['geomean_host_ms']
 rows=[];sources=[]
 for source in catalog:
  cid=source['id'];result=summary['results'][cid];impl=result['implementations']
  directory=RUN/'artifacts'/f'final-001-diagnostics-{cid}'
  mode='dense_policy' if cid=='c00' else 'sparse_c'
  structure_path=directory/f'structure-{mode}.json';structure=read(structure_path)
  diagnostic_path=directory/f'diagnostics-{mode}.json';diagnostic=read(diagnostic_path)
  assert structure['coverage']['blocks']==338 and diagnostic['coverage']['blocks']==338
  pooled={}
  for site in ('h','z'):
   layers=[d for key,d in structure['per_site_layer'].items() if key.startswith(site+'.')]
   total=sum(d['total'] for d in layers);zeros=sum(d['exact_zero'] for d in layers)
   finite=sum(d['finite'] for d in layers);sum_squares=math.fsum(d['sum_squares'] for d in layers)
   hist=[sum(d['row_nnz_hist'][i] for d in layers) for i in range(len(layers[0]['row_nnz_hist']))]
   assert total-zeros==sum(i*n for i,n in enumerate(hist))
   pooled[site]={'total':total,'exact_zero':zeros,'zero_fraction':zeros/total,'finite':finite,'sum_squares':sum_squares,'rms':math.sqrt(sum_squares/finite),'row_nnz_hist':hist}
  for path in (structure_path,diagnostic_path):sources.append({'path':path.relative_to(RUN).as_posix(),'sha256':sha(path)})
  primary=impl['sparse_c_graph'];new_ms=primary['geomean_host_ms']
  row={'id':cid,'recipe':'Base' if cid=='c00' else 'T2/Ph' if source['family']=='HZ+OL1@h' else 'T7/Pall','kappa':source['dose'],
       'native_BF16_loss':result['native_loss'][0],'kernel_BF16_loss':primary['process_loss'][0],
       'canonical_FP16_loss':source['canonical_logical_products']['coverage']['loss'],
       'canonical_R_model_fraction':source['canonical_logical_products']['measured']['R_model'],
       'latency_ms':{k:v['geomean_host_ms'] for k,v in impl.items()},
       'qualified':{k:v['qualified'] for k,v in impl.items()},
       'new_kernel_speedup_vs_pytorch_Base':base_pt/new_ms,'new_kernel_speedup_vs_dense_Base':base_dense/new_ms,
       'new_kernel_speedup_vs_own_dense_policy':impl['dense_policy_graph']['geomean_host_ms']/new_ms,
       'new_kernel_speedup_vs_opt073':impl['opt073_graph']['geomean_host_ms']/new_ms,
       'paired_sparse_vs_best_dense':result['contrasts']['sparse_c_graph'],
       'pooled_BF16_activations':pooled,'checkpoint_weight_sha256':next(r['sha256'] for r in source['files'] if r['path'].endswith('model.safetensors'))}
  rows.append(row)
 write(RUN/'results/complete-table.json',{'selection_sha256':summary['selection_sha256'],'coverage':summary['coverage'],'rows':rows,'diagnostic_sources':sources,
       'denominators':{'pytorch_Base_ms':base_pt,'optimized_dense_Base_ms':base_dense},
       'limits':summary['limits']})
 lines=['# Complete 70M frozen-kernel table','',
        'RTX5090, BF16, batch1, 2048 tokens, full50304 logits, CUDA graphs. Each latency is the geometric mean of1344 synchronized host timings across three fresh processes. Loss is native BF16 full-validation loss; canonical historical FP16 loss is retained separately in the JSON. Speedup above1 means faster.','',
        '| Reference | Loss | Latency (ms) |', '| --- | ---: | ---: |',
        f'| Base PyTorch | {rows[0]["native_BF16_loss"]:.4f} | {base_pt:.6f} |',
        f'| Base optimized dense | {rows[0]["native_BF16_loss"]:.4f} | {base_dense:.6f} |','',
        '| Recipe | kappa | Loss | PyTorch (ms) | Dense h/z control (ms) | Previous opt073 (ms) | New kernel (ms) | x PyTorch Base | x Dense Base | Qualified |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |']
 for row in rows:
  if row['id']=='c00':continue
  latency=row['latency_ms']
  lines.append(f'| {row["recipe"]} | {row["kappa"]:g} | {row["native_BF16_loss"]:.4f} | {latency["native_graph"]:.6f} | {latency["dense_policy_graph"]:.6f} | {latency["opt073_graph"]:.6f} | {latency["sparse_c_graph"]:.6f} | {row["new_kernel_speedup_vs_pytorch_Base"]:.3f} | {row["new_kernel_speedup_vs_dense_Base"]:.3f} | {"yes" if row["qualified"]["sparse_c_graph"] else "NO"} |')
 lines += ['', 'The dense h/z control uses the same checkpoint and unchanged non-h/z scaffold. For T7, that scaffold retains its earlier gate-aware execution at other sites; this column is not an all-sites no-skip ablation. The optimized dense Base has no active gates.','',
   'Compared with opt073, the new policy compacts active h-feature unions over small token groups and gathers only the corresponding weight rows for tensor-core multiplication. It sparsifies h in five layers, retains dense h in layer0 and dense gated z everywhere, and fuses gates/output combination to reduce overhead. It does not impose the old short-row nonzero cutoff and introduces no extra pruning. The same frozen layer policy is used at every kappa and for both topologies.','',
   'Qualification covers all338 complete validation blocks from500 documents (692224 input tokens,691886 prediction tokens;1444-token tail excluded): each logit within .25+.02*abs(reference), relativeL2<=.02, pooled loss difference<=.001. Numerical failures, if any, are not usable speedup claims. Base ratios compare separate fresh processes within this retained session; same-checkpoint dense/sparse timings are paired.','',
   'Source: [final-summary.json](final-summary.json), [complete-table.json](complete-table.json); generated by [06_table.py](../06_table.py).', '',
   '| Recipe | kappa | h exact zeros (%) | z exact zeros (%) | Sparse / own dense speedup | Paired crossed-cluster95% interval |',
   '| --- | ---: | ---: | ---: | ---: | --- |']
 for row in rows[1:]:
  p=row['pooled_BF16_activations'];ci=row['paired_sparse_vs_best_dense']['crossed_process_input_95']
  lines.append(f'| {row["recipe"]} | {row["kappa"]:g} | {100*p["h"]["zero_fraction"]:.4f} | {100*p["z"]["zero_fraction"]:.4f} | {row["new_kernel_speedup_vs_own_dense_policy"]:.4f} | {ci[0]:.4f}--{ci[1]:.4f} (vs best qualified dense h/z control) |')
 (RUN/'results/complete-table.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
 print('\n'.join(lines[:25]))

if __name__=='__main__':main()

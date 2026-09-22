"""Reduce retained full-validation structure without changing frozen policies."""
import math,re
from support import RUN,read,write,sha

def main():
 selection=read(RUN/'provenance/selection-final.json');results={};sources=[]
 for cid in ('c00','c24','c25'):
  root=RUN/'artifacts'/f'diagnostics-{cid}-001'
  result=read(root/'result.json');assert result['status']=='complete'
  modes={}
  for mode in result['implementations']:
   paths=[root/f'{kind}-{mode}.json' for kind in ('structure','work','compiler','diagnostics','profile-summary')]
   structure,work,compiler,diagnostics,profiles=map(read,paths)
   assert structure['coverage']['blocks']==338
   layers={};pooled={}
   for key,x in structure['per_site_layer'].items():
    site,index=key.split('.');hist=x['row_nnz_hist'];rows=sum(hist);nnz=sum(i*n for i,n in enumerate(hist));assert x['total']-x['exact_zero']==nnz
    row={'exact_zero_fraction':x['exact_zero']/x['total'],'rms':x['rms'],'mean_nnz_per_row':nnz/rows,
         'maximum_row_nnz':max(i for i,n in enumerate(hist) if n),
         'fraction_rows_nnz_at_most':{str(limit):sum(hist[:limit+1])/rows for limit in (8,16,32,64)},
         'fraction_four_feature_groups_at_most_two':sum(x['four_feature_nnz_hist'][:3])/sum(x['four_feature_nnz_hist'])}
    segment=x['segment256_nnz_hist']
    row['segment256']={'segments':sum(segment),'maximum_nnz':max(i for i,n in enumerate(segment) if n),
      'overflow_counts':{str(cap):{'segments_exceeding_capacity':sum(segment[cap+1:]),'excess_values':sum(max(i-cap,0)*n for i,n in enumerate(segment))} for cap in (31,63,127)}}
    row['exact24_eligible_tiles']={f'M{bm}_K{bk}':{'eligible':x[f'tiles24_M{bm}_K{bk}']-x[f'overflow24_M{bm}_K{bk}'],'total':x[f'tiles24_M{bm}_K{bk}']} for bm in (16,32,64) for bk in (32,64)}
    for gm in (1,4,8,16,32):
     union=x[f'union{gm}_hist'];row[f'mean_union_{gm}']=sum(i*n for i,n in enumerate(union))/sum(union)
    w=work['per_site_layer'][key];row['backend']=w['backend']
    if w['backend'].startswith('c_'):
     stem=f'{site}-{index}-1';entry=compiler[stem];spec=entry['backend'];bn=spec.get('bn',64);bk=spec['bk']
     ptxpath=root/('compiler-'+mode)/(stem+'.ptx');ptx=ptxpath.read_text();mma=re.findall(r'\bmma\.sync\.aligned\.m16n8k16\.row\.col\.f32\.bf16\.bf16\.f32\b',ptx)
     # Every warp traverses one uniform K loop. This equality also catches an
     # unexpected unrolling/layout change before issuing an instruction count.
     assert len(mma)*4*16*8*16==16*bn*bk,(stem,len(mma))
     row['loop_derived_warp_mma_instances']=len(mma)*4*w['consumer_tile_iterations']
     row['mma_count_unit']='PTX warp-instruction instances,derived from checked runtime loop counts; not a hardware counter'
     row['padded_product_positions_over_dense_products']=w['padded_tensorcore_product_positions']/w['dense_products']
     row['consumer_K_iteration_bypass_fraction']=w['dense_K_iterations_bypassed']/(w['consumer_tile_iterations']+w['dense_K_iterations_bypassed'])
     row['compiler_registers']=entry['registers'];row['compiler_spills']=entry['spills']
     row['requested_weight_bytes']=2*w['weight_values_requested']
     row['conversion_input_bytes_scanned']=2*w['input_values']
     row['requested_bytes_caution']='Scalar requested bytes from the frozen source and checked metadata; not measured L2/DRAM transactions.'
     sources.append({'path':ptxpath.relative_to(RUN).as_posix(),'sha256':sha(ptxpath)})
    layers[key]=row
    group=pooled.setdefault(site,{'total':0,'exact_zero':0,'finite':0,'sum_squares':0.,'rows':0,'nnz':0})
    for name in ('total','exact_zero','finite','sum_squares'):group[name]+=x[name]
    group['rows']+=rows;group['nnz']+=nnz
   for group in pooled.values():
    group['zero_fraction']=group['exact_zero']/group['total'];group['rms']=math.sqrt(group['sum_squares']/group['finite']);group['mean_nnz_per_row']=group['nnz']/group['rows']
   kernel_profiles=[]
   for profile in profiles['profiles']:
    kernels=profile['kernels'];selected={name:v for name,v in kernels.items() if any(term in name for term in ('gathered','union_index','dense_gate','consume','pack','combine'))}
    kernel_profiles.append({'execution':profile['execution'],'inputs':profile['inputs'],'selected_kernels':selected})
   modes[mode]={'pooled':pooled,'layers':layers,'profiles':kernel_profiles,'hardware_memory_note':'Requested values and PTX-derived work are distinct from cache transactions. See hardware-001 for supported measured counters or its retained permission failure.'}
   sources.extend({'path':p.relative_to(RUN).as_posix(),'sha256':sha(p)} for p in paths)
  results[cid]=modes
 hardware=read(RUN/'artifacts/hardware-001/result.json')
 write(RUN/'results/mechanism-summary.json',{'status':'complete','reduction_source_sha256':sha(RUN/'13_mechanism.py'),'precision':'actual BF16 operands','selection_sha256':sha(RUN/'provenance/selection-final.json'),'results':results,'hardware_counter_status':hardware,'sources':sources})
 print({'hardware':hardware['status'],'pooled_primary':{c:r[selection['primary'] if c!='c00' else 'dense_policy']['pooled'] for c,r in results.items()}})

if __name__=='__main__':main()

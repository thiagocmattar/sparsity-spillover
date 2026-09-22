"""Runtime metadata checks and requested work, distinct from hardware traffic."""
import torch

class Work:
 def __init__(self):self.rows={}
 def add(self,key,op,x):
  m,k=x.shape;n=512
  row=self.rows.setdefault(key,{'blocks':0,'input_values':0,'active_values':0,'dense_products':0,'nonzero_activation_products':0})
  def add(name,value):row[name]=row.get(name,0)+int(value)
  active=x!=0;nnz=int(active.sum());add('blocks',1);add('input_values',x.numel());add('active_values',nnz)
  add('dense_products',m*k*n);add('nonzero_activation_products',nnz*n)
  if op is None:row['backend']='native';return
  spec=op.spec;f=spec['family'];row['backend']=spec['id']
  if f=='a':
   expected=active.reshape(m,k//256,256).sum(-1).int()
   assert torch.equal(expected,op.count),'Actual packed counts disagree with operands'
   add('packed_values_written',int(op.count.sum()));add('segment_count_words',op.count.numel())
   # One warp owns 32*V outputs. All valid packed values are consumed once per
   # output tile; unlike a byte-traffic counter this ignores cache transactions.
   tiles=(n+32*spec['v']-1)//(32*spec['v'])
   add('consumer_warp_chunks',int(((op.count+31)//32).sum())*tiles)
   add('consumer_packed_words_requested',nnz*tiles)
   add('weight_values_requested',nnz*n);add('scalar_products_executed',nnz*n)
  elif f=='c':
   gm,bk,bn=spec['gm'],spec['bk'],spec.get('bn',64)
   unions=active.reshape(m//gm,gm,k).any(1);expected=unions.sum(-1).int()
   assert torch.equal(expected,op.count),'Actual union counts disagree with operands'
   # Validate active index lists, including changed support; ignored storage
   # past count is intentionally unspecified and never consumed by the kernel.
   positions=torch.arange(k,device=x.device)[None,:]<op.count[:,None]
   idx=op.idx.long();valid=idx[positions]
   group=torch.arange(m//gm,device=x.device)[:,None].expand_as(idx)[positions]
   reconstructed=torch.zeros_like(unions);reconstructed[group,valid]=True
   assert torch.equal(reconstructed,unions),'Union indices lost or added features'
   metadata_count=int(op.count.sum());count=metadata_count;iterations=int(((op.count+bk-1)//bk).sum());nt=(n+bn-1)//bn
   if spec.get('no_skip',False):count=(m//gm)*k;iterations=(m//gm)*((k+bk-1)//bk)
   add('union_indices_written',metadata_count);add('consumer_indices_used',count);add('consumer_tile_iterations',iterations*nt)
   add('weight_values_requested',count*n);add('activation_values_requested',count*gm*nt)
   add('padded_tensorcore_product_positions',iterations*bk*16*n)
   add('products_in_row_padding',iterations*bk*(16-gm)*n)
   add('dense_K_iterations_bypassed',((m//gm)*((k+bk-1)//bk)-iterations)*nt)
  elif f=='d':
   first=x-op.e;assert torch.equal(first+op.e,x)
   assert int((first.reshape(m,k//4,4)!=0).sum(-1).max())<=2
   bm,bk=spec['bm'],spec['bk'];tiles=(op.e!=0).reshape(m//bm,bm,k//bk,bk).any(3).any(1)
   add('compressed_values_written',m*k//2);add('metadata_words_written',m*k//16)
   add('overflow_values',int((op.e!=0).sum()))
   add('overflow_dense_tile_iterations',int(tiles.sum())*(n//64))
   add('overflow_empty_tile_bypasses',int((~tiles).sum())*(n//64))
  elif f=='dense':row['issued_work']='not measured; dense logical products above are not instruction counts'
 def result(self):
  return {'method':'Counts read from actual conversion buffers and checked against independent operand masks. Consumer iterations and requested values follow the frozen kernel loops; these are not hardware memory transactions.',
          'hardware_dram_cache_counters':'see separate hardware-counter attempts; not inferred from these counts',
          'mma_instruction_counts':'not measured directly; padded product positions are reported without relabeling them as issued instructions',
          'per_site_layer':self.rows}

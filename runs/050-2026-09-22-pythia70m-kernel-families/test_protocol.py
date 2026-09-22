"""Focused conservation, threshold and coverage checks, without GPU imports."""
import importlib.util
from pathlib import Path
import numpy as np
import torch

RUN=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('run050_structure',RUN/'structure.py')
structure=importlib.util.module_from_spec(spec);spec.loader.exec_module(structure)

def test_equality_and_bf16_threshold():
 x=torch.tensor([-.1,0,.099,.1,.101],dtype=torch.bfloat16)
 y=structure.gated(x,.1)
 assert torch.equal(y[:3],torch.zeros(3,dtype=torch.bfloat16))
 assert torch.equal(y[3:],x[3:])

def test_counts_pool_integers_and_overflow():
 x=torch.zeros(64,512,dtype=torch.bfloat16);x[:,0:3]=1
 s=structure.Structure();s.add('h',x);s.add('h',x)
 d=s.result()['h']
 assert d['row_nnz_hist'][3]==128 and sum(d['row_nnz_hist'])==128
 assert d['exact_zero']==2*64*509 and d['total']==2*x.numel()
 assert d['four_feature_nnz_hist'][3]==128
 assert d['overflow24_M16_K32']==8
 assert sum(d['segment256_nnz_hist'])==256
 assert d['union16_hist'][3]==8
 assert d['sum_squares']==384 and abs(d['rms']-(3/512)**.5)<1e-12

def test_training_split_disjoint_and_literal():
 import json,hashlib
 manifest=json.loads((RUN/'provenance/inputs.json').read_text())
 a=manifest['development_indices'];b=manifest['confirmation_indices']
 assert a==list(range(64)) and b==list(range(64,128)) and not set(a)&set(b)
 raw=(RUN/manifest['training']['path']).read_bytes()
 assert len(raw)==128*2048*4 and hashlib.sha256(raw).hexdigest()==manifest['training']['sha256']
 source=RUN.parents[1]/manifest['training_source_path']
 with source.open('rb') as f:assert f.read(len(raw))==raw

def test_selection_uses_within_screen_ratios_and_both_kappas():
 import sys,copy
 sys.path.insert(0,str(RUN))
 spec=importlib.util.spec_from_file_location('run050_selection',RUN/'04_select.py')
 selector=importlib.util.module_from_spec(spec);spec.loader.exec_module(selector)
 screens=[]
 for scale in (1,20):
  screen={}
  for cid in ('c24','c25'):
   for split in ('development','confirmation'):
    for site in ('h','z'):
     for i in range(6):
      screen[f'{cid}:{split}:{site}.{i}']={name:{'host_ms':scale*t,'qualified':True} for name,t in [('dense_native',1.2),('dense_fused',1.),('dense_dot',1.1),('c_good',.8),('a_one_kappa',.7 if cid=='c24' else 1.1)]}
  screens.append(screen)
 selected=selector.select(screens)
 assert set(selected['dense'].values())=={'dense_fused'}
 assert set(selected['policies'])=={'sparse_c'}
 assert selected['policies']['sparse_c']['h.0']=='c_good'
 assert selected['ablations']['sparse_c_no_skip']['h.0']=='c_good_noskip'
 screens[1]['c25:confirmation:h.0']['c_good']['qualified']=False
 assert selector.select(screens)['policies']['sparse_c']['h.0']=='dense_fused'

def test_runtime_union_counts_are_checked_and_padding_is_separate():
 import sys
 from types import SimpleNamespace
 sys.path.insert(0,str(RUN))
 from work_counters import Work
 x=torch.zeros(64,512,dtype=torch.bfloat16);x[:,:3]=1
 idx=torch.zeros(16,512,dtype=torch.int32);idx[:,:3]=torch.arange(3)
 op=SimpleNamespace(spec={'id':'c_test','family':'c','gm':4,'bk':32},count=torch.full((16,),3,dtype=torch.int32),idx=idx)
 work=Work();work.add('h.0',op,x);row=work.result()['per_site_layer']['h.0']
 assert row['consumer_tile_iterations']==16*8
 assert row['weight_values_requested']==16*3*512
 assert row['nonzero_activation_products']==64*3*512
 assert row['products_in_row_padding']==16*32*12*512
 op.spec['no_skip']=True;dense=Work();dense.add('h.0',op,x)
 assert dense.result()['per_site_layer']['h.0']['dense_K_iterations_bypassed']==0
 op.spec['no_skip']=False
 op.count[0]=2
 import pytest
 with pytest.raises(AssertionError):work.add('h.0',op,x)

def test_paired_reducer_preserves_exact_effect_and_rejects_duplicates():
 import sys,math
 sys.path.insert(0,str(RUN))
 spec=importlib.util.spec_from_file_location('run050_reducer',RUN/'11_reduce.py');reducer=importlib.util.module_from_spec(spec);spec.loader.exec_module(reducer)
 trials=[]
 for process in range(3):
  samples=[]
  for repeat in range(7):
   for index in range(64):
    t=1.+index/100+process/10
    samples.extend([{'repeat':repeat,'input_index':index,'mode':'dense','host_ms':2*t},{'repeat':repeat,'input_index':index,'mode':'sparse','host_ms':t}])
  trials.append({'samples':samples})
 result=reducer.paired_interval(trials,'dense','sparse',draws=100)
 assert abs(result['speedup']-2)<1e-12
 assert all(abs(v-2)<1e-12 for v in result['stratified_input_block_95'])
 assert all(abs(v-2)<1e-12 for v in result['crossed_process_input_95'])
 trials[0]['samples'].append(trials[0]['samples'][0])
 import pytest
 with pytest.raises(AssertionError):reducer.paired_interval(trials,'dense','sparse',draws=1)

def test_bootstrap_preserves_repeated_input_correlation():
 import sys,math
 sys.path.insert(0,str(RUN))
 spec=importlib.util.spec_from_file_location('run050_reducer_correlation',RUN/'11_reduce.py');reducer=importlib.util.module_from_spec(spec);spec.loader.exec_module(reducer)
 samples=[]
 for repeat in range(7):
  for index in range(64):
   samples.extend([{'repeat':repeat,'input_index':index,'mode':'dense','host_ms':math.exp(index/100)},
                   {'repeat':repeat,'input_index':index,'mode':'sparse','host_ms':1.}])
 result=reducer.paired_interval([{'samples':samples} for _ in range(3)],'dense','sparse',draws=1000)
 independent=result['stratified_input_block_95'];shared=result['shared_input_block_95']
 assert shared[1]-shared[0]>1.4*(independent[1]-independent[0])

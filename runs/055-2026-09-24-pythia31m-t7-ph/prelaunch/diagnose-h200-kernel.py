"""One-off component substitution diagnostic; never latency evidence."""
from pathlib import Path
import subprocess

WORKER = r'''
import pathlib,sys,json,os,traceback
import numpy as np
import torch,transformers
r=pathlib.Path('/workspace/sparsity-spillover/runs/055-2026-09-24-pythia31m-t7-ph')
c=pathlib.Path('/workspace/run055-control')
sys.path.insert(0,str(r/'latency'))
import port
from sparsity_research.pythia import load_checkpoint_pythia
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
row,=[v for p in (r/'artifacts').glob('verification-*.json') for v in json.loads(p.read_text()).get('conditions',[]) if v['condition']['id']=='a7-h-ol1-kappa-0' and v['status']=='verified']
checkpoint=r/'artifacts/attempts'/row['attempt']/row['final_checkpoint']['path']
data=np.memmap(r.parents[1]/'data/tokenized/minipile-pythia-14m-full/validation/tokens.int32.bin',dtype=np.int32,mode='r')
def load():
 m=load_checkpoint_pythia(transformers.AutoModelForCausalLM,checkpoint,torch=torch).to('cuda',dtype=torch.bfloat16).eval()
 m.set_attn_implementation('sdpa');m.config.use_cache=False
 return m
def compare(a,b):
 a=a.float();b=b.float();d=(a-b).abs();bound=.25+.02*a.abs()
 return dict(max_abs=float(d.max()),relative_l2=float(torch.linalg.vector_norm(d)/torch.linalg.vector_norm(a)),bad_elements=int((d>bound).sum()))
out=dict(scientific_measurement=False,scope='component substitution on failed block15',rows=[])
try:
 with torch.inference_mode():
  ids=torch.tensor(data[15*2048:16*2048].copy(),device='cuda',dtype=torch.long)[None]
  native=load();reference=native(input_ids=ids,use_cache=False).logits.clone()
  del native
  for variant in ['kernel','native_norm','native_attention','native_joint','native_head']:
   m=load();old=[]
   for layer in m.gpt_neox.layers:
    old.append(dict(norm1=layer.input_layernorm.forward,norm2=layer.post_attention_layernorm.forward,a=layer.a_gate.forward,m=layer.m_gate.forward,attention=layer.attention.forward,layer=layer.forward,h=layer.mlp.act.forward,z=layer.attention.z_gate.forward,w2=layer.mlp.dense_4h_to_h,wo=layer.attention.dense))
   head=m.embed_out.forward
   port.install(m)
   for layer,saved in zip(m.gpt_neox.layers,old):
    if variant=='native_norm':
     layer.input_layernorm.forward=saved['norm1'];layer.post_attention_layernorm.forward=saved['norm2'];layer.a_gate.forward=saved['a'];layer.m_gate.forward=saved['m']
    if variant=='native_attention':layer.attention.forward=saved['attention']
    if variant=='native_joint':
     layer.forward=saved['layer'];layer.mlp.act.forward=saved['h'];layer.attention.z_gate.forward=saved['z'];layer.mlp.dense_4h_to_h=saved['w2'];layer.attention.dense=saved['wo']
   if variant=='native_head':m.embed_out.forward=head
   result=port.replay.scaffold.forward(m,ids).clone()
   row=dict(variant=variant,**compare(reference,result));out['rows'].append(row)
   print(json.dumps(row),flush=True)
   del m,result,old,head
   torch.cuda.empty_cache()
 out['status']='completed'
except BaseException as e:
 out.update(status='failed',error=str(e),traceback=traceback.format_exc());raise
finally:
 (r/'artifacts/preflight-h200-diagnosis-001.json').write_text(json.dumps(out,indent=2))
 (c/'h200-diagnosis-exit-001.json').write_text(json.dumps(out,indent=2))
'''
compile(WORKER,'h200-diagnosis-worker','exec')
CODE='WORKER='+repr(WORKER)+'\n'+r'''
import pathlib,subprocess,json,os
c=pathlib.Path('/workspace/run055-control');script=c/'h200-diagnosis-001.py';script.write_text(WORKER)
env=dict(os.environ,PATH='/workspace/run055-venv/bin:'+os.environ['PATH'],CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
with (c/'h200-diagnosis-001.log').open('x') as f:
 p=subprocess.Popen(['/workspace/run055-venv/bin/python',str(script)],env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps(dict(pid=p.pid,scientific_measurement=False)))
'''
ssh=['ssh','-i',str(Path.home()/'.runpod/ssh/runpodctl-ssh-key'),'-o','BatchMode=yes','-p','27548','root@103.196.86.181']
p=subprocess.run(ssh+['python3 -'],input=CODE,text=True,capture_output=True,check=True)
(Path(__file__).parent/'h200-diagnosis-launch-001.json').write_text(p.stdout)
print(p.stdout)

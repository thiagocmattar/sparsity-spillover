"""Freeze the requested grid; copy kernel bytes without changing Run050."""
from pathlib import Path
import hashlib,json,shutil

RUN=Path(__file__).resolve().parent
REPO=RUN.parents[1]
PREVIOUS=next((REPO/'runs').glob('050-*'))
IDS=['c00','c22','c23','c24','c25','c26','c07','c08','c09','c10','c11']
KERNELS=['primitives.py','adapter.py','structure.py','work_counters.py','packed.cu','structured.py','structured.cu']
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def write(p,d):
 p.parent.mkdir(parents=True,exist_ok=True)
 p.write_text(json.dumps(d,indent=2)+'\n',encoding='utf-8',newline='\n')
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def replace(text,old,new):
 assert old in text,old
 return text.replace(old,new)

def main():
 assert not (RUN/'provenance/inputs.json').exists(),'Already prepared'
 sources={n:next((REPO/'runs').glob(f'{n:03d}-*')) for n in (45,47)}
 manifests={n:read(p/'provenance/inputs.json') for n,p in sources.items()}
 rows=[];transfer=[]
 for cid in IDS:
  number=47 if cid=='c26' else 45
  row=next(c for c in manifests[number]['checkpoints'] if c['id']==cid)
  row={**row,'catalog_source':sources[number].relative_to(REPO).as_posix()}
  for item in row['files']+row['provenance']:
   source=sources[number]/item['path']
   assert source.stat().st_size==item['bytes'] and sha(source)==item['sha256'],str(source)
   transfer.append({**item,'source':source.relative_to(REPO).as_posix(),'reuse_run049':cid in ('c00','c24','c25')})
   if item in row['provenance']:
    dest=RUN/item['path'];dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,dest)
  rows.append(row)
 write(RUN/'provenance/inputs.json',{'checkpoints':rows,'validation':manifests[45]['validation'],'validation_metadata':manifests[45]['validation_metadata']})
 write(RUN/'provenance/transfer-inputs.json',{'files':transfer})
 for name in KERNELS+['support.py']:
  shutil.copyfile(PREVIOUS/name,RUN/name)
 selection=read(PREVIOUS/'provenance/selection-final.json')
 shutil.copyfile(PREVIOUS/'provenance/selection-final.json',RUN/'provenance/selection-final.json')
 write(RUN/'provenance/kernel-reuse.json',{'run050_selection_sha256':sha(PREVIOUS/'provenance/selection-final.json'),'files':[{'path':n,'sha256':sha(RUN/n),'bytes':(RUN/n).stat().st_size} for n in KERNELS]})
 cfg={k:read(PREVIOUS/'config.json')[k] for k in ('validation_blocks','timing_inputs','timing_passes','timing_seed','process_replicates','deadline_utc','numerical_bounds')}
 cfg.update(conditions=IDS,implementations=['native','native_hz','opt073','dense_policy','sparse_c'],selection='Run050 frozen policy; no tuning on this grid',authorization='User requested the complete frozen-kernel table after approving launch and the retained GPU extension.')
 write(RUN/'config.json',cfg)
 text=(PREVIOUS/'bootstrap.py').read_text()
 text=replace(text,"read(BASE/'provenance/inputs.json')['checkpoints']","read(RUN/'provenance/inputs.json')['checkpoints']")
 text=replace(text,'base_io.verify(item)','base_io.verify(item,root=RUN)')
 text=replace(text,"BASE/row['checkpoint']","RUN/row['checkpoint']")
 text += '\n\ndef verify_frozen():\n    parent=__import__("pathlib").Path(os.environ.get("RUN051_PREVIOUS", str(RUN.parent/"050-2026-09-22-pythia70m-kernel-families")))\n    original=read(parent/"provenance/selection-final.json")\n    assert sha(RUN/"provenance/selection-final.json")==sha(parent/"provenance/selection-final.json")\n    for name,digest in original["kernel_source_hashes"].items():assert sha(parent/name)==digest,name\n    for item in read(RUN/"provenance/kernel-reuse.json")["files"]:assert sha(RUN/item["path"])==item["sha256"],item["path"]\n    for item in read(RUN/"provenance/source-freeze.json")["files"]:assert sha(RUN/item["path"])==item["sha256"],item["path"]\n'
 (RUN/'bootstrap.py').write_text(text,encoding='utf-8',newline='\n')
 text=(PREVIOUS/'05_benchmark.py').read_text()
 text=replace(text,"choices=('c00','c24','c25')","choices=read(RUN/'config.json')['conditions']")
 text=replace(text,"   for name,digest in selection['kernel_source_hashes'].items():assert sha(RUN/name)==digest,('Frozen source changed',name)","   bootstrap.verify_frozen()")
 # This grid evaluates only the frozen winner and its controls, with no search.
 start=text.index("  policy_modes={'dense_fused'")
 end=text.index("  modes=['native'",start)
 text=text[:start]+"  policy_modes={'dense_policy':selection['dense'],'sparse_c':selection['policies']['sparse_c']}\n  result.update(policies=policy_modes)\n"+text[end:]
 text=replace(text,"read(RUN/'provenance/inputs.json')['training'];path=RUN/manifest['path'];assert sha(path)==manifest['sha256']","read(BASE/'provenance/inputs.json')['validation'];path=bootstrap.base_io.verify(manifest)")
 (RUN/'02_benchmark.py').write_text(text,encoding='utf-8',newline='\n')
 text=(PREVIOUS/'07_diagnostics.py').read_text()
 text=replace(text,"choices=('c00','c24','c25')","choices=read(RUN/'config.json')['conditions']")
 text=replace(text,"  for name,digest in selection['kernel_source_hashes'].items():assert sha(RUN/name)==digest","  bootstrap.verify_frozen()")
 text=replace(text,"read(RUN/'provenance/inputs.json')['training'];path=RUN/manifest['path'];assert sha(path)==manifest['sha256']","read(BASE/'provenance/inputs.json')['validation'];path=bootstrap.base_io.verify(manifest)")
 text=replace(text," policies={'dense_policy':selection['dense'],**selection['policies'],**selection.get('ablations',{})}"," policies={'dense_policy':selection['dense'],'sparse_c':selection['policies']['sparse_c']}")
 (RUN/'03_diagnostics.py').write_text(text,encoding='utf-8',newline='\n')
 (RUN/'prelaunch').mkdir(exist_ok=True)
 for name in ('transport.py','start.py','status.py','fetch_bundle.py','audit_retrieval.py','ssh_read.py'):
  text=(PREVIOUS/'prelaunch'/name).read_text().replace('/workspace/run050','/workspace/run051')
  if name=='start.py':text=replace(text,"'export RUN050_BASE=/workspace/run049',","'export RUN050_BASE=/workspace/run049 RUN051_PREVIOUS=/workspace/run050',")
  (RUN/'prelaunch'/name).write_text(text,encoding='utf-8',newline='\n')
 print({'conditions':IDS,'kernel_files_unchanged':len(KERNELS),'transfer_bytes':sum(x['bytes'] for x in transfer if not x['reuse_run049'])})

if __name__=='__main__':main()

"""Freeze original source and all retained 70M inference inputs; never retrain."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tarfile

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
BASE=ROOT/'runs/033-2026-09-17-a7-h-only-k050-benchmark'

def fs(path):
    path=Path(path).absolute()
    return Path('\\\\?\\'+str(path)) if os.name=='nt' and not str(path).startswith('\\\\?\\') else path

def sha(path):
    with fs(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def record(path,root=HERE):
    return {'path':path.relative_to(root).as_posix(),'bytes':fs(path).stat().st_size,'sha256':sha(path)}

def read(path):return json.loads(fs(path).read_text(encoding='utf-8'))

def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n',encoding='utf-8',newline='\n')

def prepare():
    if (HERE/'artifacts/attempts').exists():raise RuntimeError('Inputs immutable after execution')
    copied=[]
    def copy(source,relative,expected=None):
        target=HERE/relative
        if expected:assert record(source,BASE)==expected,source
        fs(target.parent).mkdir(parents=True,exist_ok=True)
        if not fs(target).exists() or sha(source)!=sha(target):shutil.copyfile(fs(source),fs(target))
        copied.append({'source':record(source,ROOT),'copy':record(target)})
        return record(target)
    archive=read(BASE/'provenance/archive.json')
    for row in archive['files']:
        item=row['snapshot'];copy(BASE/item['path'],item['path'],item)
    write(HERE/'provenance/archive.json',archive)
    copy(BASE/'provenance/pip-freeze.txt','provenance/pip-freeze.txt')
    baseline=read(BASE/'provenance/inputs.json')
    validation=copy(BASE/baseline['validation']['path'],'inputs/validation.int32.bin',baseline['validation'])
    checkpoints=[];sources=[];identities=set()
    for prefix in ['018-','034-']:
        train=next((ROOT/'runs').glob(prefix+'*'))
        cohort=read(train/'artifacts/verification.json')
        assert cohort['status']=='verified' and cohort['evidence_label']=='valid'
        sources.append(record(train/'artifacts/verification.json',ROOT))
        for v in cohort['conditions']:
            c=v['condition'];source=train/'artifacts/attempts'/v['attempt_id']
            checkpoint=source/'checkpoints/step_000712';identifier=f'c{len(checkpoints):02d}'
            manifest=read(source/'manifest.json');logical=read(source/'diagnostics/logical_products.json')
            assert manifest['completed_steps']==712 and manifest['input_tokens']==1493172224
            identities.add((manifest['initial_parameter_sha256'],manifest['training_schedule_hash']))
            cfg=read(checkpoint/'config.json')
            assert (cfg['hidden_size'],cfg['intermediate_size'],cfg['num_attention_heads'])==(512,2048,8)
            names=['config.json','model.safetensors','checkpoint_metadata.json','generation_config.json']
            files=[copy(checkpoint/n,f'inputs/checkpoints/{identifier}/{n}') for n in names]
            provenance=[copy(source/n,f'provenance/originals/{identifier}/{n}') for n in
                        ['manifest.json','config.yaml','diagnostics/logical_products.json']]
            top=c['topology_id']
            family=top if top in {'A0','A1-H'} else ('A4' if top=='A4-Z' else 'A7')+'+OL1@'+('h' if c['pressure_sites']==['h'] else 'all')
            checkpoints.append({'id':identifier,'family':family,'dose':c['gate_threshold'],
                'checkpoint':f'inputs/checkpoints/{identifier}','files':files,'provenance':provenance,
                'source':source.relative_to(ROOT).as_posix(),'source_condition':c,
                'original_checkpoint':checkpoint.relative_to(ROOT).as_posix(),
                'original_files':[record(checkpoint/n,ROOT) for n in names],
                'topology':cfg,'canonical_logical_products':logical,
                'final_checkpoint_content_sha256':v['checkpoint_content_sha256']})
    assert len(checkpoints)==22 and len(identities)==1
    write(HERE/'provenance/inputs.json',{'checkpoints':checkpoints,'validation':validation,
        'validation_metadata':baseline['validation_metadata'],'source_manifests':sources})
    write(HERE/'provenance/reuse.json',{'base_run':BASE.relative_to(ROOT).as_posix(),'files':copied})
    write(HERE/'provenance/candidates.json',{'configurations':[{'id':'k050-70m-v1','candidate':'k050-70m-v1',
        'status':'eligible','settings':{'round_p':False,'shortcut':False,'skip':True,'projection_skip':True}}]})
    print(f'Prepared {len(checkpoints)} final checkpoints and {len(archive["files"])} frozen sources',flush=True)

def verify():
    inputs=read(HERE/'provenance/inputs.json')
    rows=[inputs['validation']]+[r['snapshot'] for r in read(HERE/'provenance/archive.json')['files']]
    rows += [f for c in inputs['checkpoints'] for f in c['files']+c['provenance']]
    for row in rows:
        path=HERE/row['path'];assert path.resolve().is_relative_to(HERE)
        assert record(path)==row,path
    print(f'Verified {len(rows)} frozen source/input identities',flush=True)

def bundle(tag):
    verify();target=HERE/'bundles'/f'input-{tag}.tar';target.parent.mkdir(exist_ok=True)
    assert not target.exists()
    paths=list(HERE.glob('*.py'))+list(HERE.glob('*.sh'))+[HERE/'config.json']
    paths += [HERE/r['copy']['path'] for r in read(HERE/'provenance/reuse.json')['files']]
    paths += list((HERE/'kernel').rglob('*'))+list((HERE/'provenance').glob('*'))
    paths=sorted({p for p in paths if fs(p).is_file() and '__pycache__' not in p.parts})
    write(HERE/'bundles'/f'input-{tag}-inventory.json',{'files':[record(p) for p in paths]})
    with tarfile.open(target,'w') as f:
        for p in paths:f.add(fs(p),arcname=p.relative_to(HERE).as_posix(),recursive=False)
    write(target.with_suffix('.receipt.json'),record(target));print(record(target),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','verify','bundle']);p.add_argument('--tag',default='001');a=p.parse_args()
    if a.action=='bundle':bundle(a.tag)
    else:globals()[a.action]()

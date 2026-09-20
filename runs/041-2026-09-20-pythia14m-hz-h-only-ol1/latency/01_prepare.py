"""Attach exactly the four verified Run041 endpoints to frozen K050 inputs."""
import argparse
from pathlib import Path
import shutil
import tarfile
from io_utils import RUN, read, write, record, verify, fs


def prepare():
    source=RUN.parent
    cohort=read(source/'artifacts/verification.json')
    if cohort['status']!='verified' or cohort['condition_count']!=4:
        raise ValueError('Four verified training conditions required')
    checkpoints=[]
    for index,row in enumerate(sorted(cohort['conditions'],key=lambda x:x['condition']['gate_threshold'])):
        attempt=source/'artifacts/attempts'/row['attempt_id']
        checkpoint=attempt/'checkpoints/step_000712';identifier=f'c{index:02d}'
        files=[];provenance=[]
        for name in ['config.json','model.safetensors','checkpoint_metadata.json','generation_config.json']:
            target=RUN/'inputs/checkpoints'/identifier/name;target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(checkpoint/name,target);files.append(record(target))
        for name in ['manifest.json','config.yaml','diagnostics/logical_products.json']:
            target=RUN/'provenance/originals'/identifier/name;target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(attempt/name,target);provenance.append(record(target))
        checkpoints.append({'id':identifier,'family':'HZ+OL1@h','dose':row['condition']['gate_threshold'],
            'historical':False,'checkpoint':f'inputs/checkpoints/{identifier}',
            'files':files,'provenance':provenance,'source_condition':row['condition'],
            'topology':read(checkpoint/'config.json'),
            'canonical_logical_products':read(attempt/'diagnostics/logical_products.json'),
            'final_checkpoint_content_sha256':row['checkpoint_content_sha256']})
    if [x['dose'] for x in checkpoints]!=[0,.01,.05,.1]:raise ValueError('Wrong grid')
    cache=source.parents[1]/'data/tokenized/minipile-pythia-14m-full/validation'
    target=RUN/'inputs/validation.int32.bin';shutil.copyfile(cache/'tokens.int32.bin',target)
    write(RUN/'provenance/inputs.json',{'checkpoints':checkpoints,'validation':record(target),
        'validation_metadata':read(cache/'metadata.json'),'source_manifest':record(source/'artifacts/verification.json',source)})
    verify_all()


def verify_all():
    rows=[r['snapshot'] for r in read(RUN/'provenance/archive.json')['files']]
    inputs=read(RUN/'provenance/inputs.json')
    rows+=[inputs['validation']]
    rows+=[r for c in inputs['checkpoints'] for r in c['files']+c['provenance']]
    for row in rows:verify(row)
    print(f'Verified {len(rows)} source/input identities',flush=True)


def bundle():
    verify_all();paths=list(RUN.glob('*.py'))+list(RUN.glob('*.sh'))+[RUN/'config.json']
    paths += [RUN/r['snapshot']['path'] for r in read(RUN/'provenance/archive.json')['files']]
    paths += list((RUN/'provenance').rglob('*'))+list((RUN/'inputs').rglob('*'))
    paths=sorted(set(p for p in paths if fs(p).is_file()))
    target=RUN.parent/'bundles/latency-input-001.tar.gz'
    if target.exists():raise FileExistsError(target)
    inventory={'files':[record(p) for p in paths]}
    write(RUN.parent/'bundles/latency-input-inventory.json',inventory)
    with tarfile.open(target,'w:gz',compresslevel=1) as archive:
        for path in paths:archive.add(fs(path),arcname=path.relative_to(RUN).as_posix(),recursive=False)
        archive.add(RUN.parent/'run041_topology.py',arcname='run041_topology.py',recursive=False)
    write(target.with_suffix('.receipt.json'),record(target,RUN.parent))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','verify','bundle']);a=p.parse_args()
    {'prepare':prepare,'verify':verify_all,'bundle':bundle}[a.action]()

"""Retrieve a sealed stage archive and verify every byte before integration."""
import argparse,importlib.util,shutil,sys,tarfile
from pathlib import Path
import transport

RUN=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(RUN))
from support import fs,sha,write

p=argparse.ArgumentParser();p.add_argument('tag');a=p.parse_args()
assert a.tag.replace('-','').isalnum()
dest=RUN/'retrieval'/a.tag;dest.mkdir(parents=True,exist_ok=False)
remote='/workspace/run052/'+RUN.name+'/bundles/'
stem='output-'+a.tag+'.tar'
for suffix in ('.gz','.inventory.json','.receipt.json'):
    transport.download(remote+stem+suffix,dest/(stem+suffix))
s=importlib.util.spec_from_file_location('verify_archive',RUN/'14_verify_archive.py')
v=importlib.util.module_from_spec(s);s.loader.exec_module(v)
report=v.verify(dest/(stem+'.gz'),dest/(stem+'.inventory.json'),dest/(stem+'.receipt.json'))
extracted=dest/'extracted';fs(extracted).mkdir()
with tarfile.open(dest/(stem+'.gz'),'r:gz') as archive:
    archive.extractall(fs(extracted),filter='data')
copied=0
for folder in ('artifacts','results','provenance'):
    for source in fs(extracted/folder).rglob('*'):
        if not source.is_file():continue
        target=fs(RUN/folder)/source.relative_to(fs(extracted/folder))
        if target.exists():assert sha(source)==sha(target),target
        else:
            target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target);copied+=1
report['integrated_new_files']=copied
write(RUN/'prelaunch'/('retrieval-'+a.tag+'.json'),report)
print(report)

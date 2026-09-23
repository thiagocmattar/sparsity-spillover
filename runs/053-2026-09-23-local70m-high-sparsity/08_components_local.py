"""Stage and run the bounded local component continuation; no cloud actions."""
import argparse
import io
import json
import shlex
import subprocess
import sys
import tarfile
import time
from local_support import RUN,load,write,sha

LINUX='/home/researcher/sparsity-spillover/'+RUN.name
GRID=RUN.parent/'051-2026-09-22-pythia70m-frozen-kernel-grid'
PYTHON='/opt/sparsity-gpu/venv/bin/python'

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=('stage','start','worker'))
    p.add_argument('--tag',default='components-001');args=p.parse_args()
    assert args.tag.replace('-','').isalnum()
    cfg=load(RUN/'component-config.json')
    if args.action=='stage':
        sources={name:RUN/name for name in ['wide_sparse.py','component-config.json','06_component_operators.py','07_component_screen.py','08_components_local.py']}
        sources.update({'deps/run051/'+name:GRID/name for name in ['primitives.py','support.py']})
        sources['deps/run051/selection-final.json']=GRID/'provenance/selection-final.json'
        rows=[{'path':name,'bytes':path.stat().st_size,'sha256':sha(path),'source':path.relative_to(RUN.parents[1]).as_posix()} for name,path in sources.items()]
        inventory={'files':rows,'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                   'scoped_status':subprocess.check_output(['git','status','--short','--',str(RUN)],text=True)}
        payload=io.BytesIO()
        with tarfile.open(fileobj=payload,mode='w') as archive:
            for name,path in sources.items():archive.add(path,arcname=name,recursive=False)
            data=(json.dumps(inventory,indent=2)+'\n').encode();member=tarfile.TarInfo('component-transfer.json');member.size=len(data)
            archive.addfile(member,io.BytesIO(data))
        subprocess.run(['wsl.exe','-d','SparsityGPU','--exec','tar','-xf','-','-C',LINUX],input=payload.getvalue(),check=True)
        verify='''import hashlib,json,pathlib,sys
root=pathlib.Path(sys.argv[1]);manifest=json.loads((root/'component-transfer.json').read_text())
for row in manifest['files']:
 p=root/row['path']
 with p.open('rb') as f: digest=hashlib.file_digest(f,'sha256').hexdigest()
 assert p.stat().st_size==row['bytes'] and digest==row['sha256'],str(p)
print('Verified component overlay:',len(manifest['files']))
'''
        subprocess.run(['wsl.exe','-d','SparsityGPU','--exec','python3','-',LINUX],input=verify.encode(),check=True)
        write(RUN/'results'/(args.tag+'-source.json'),inventory)
    elif args.action=='start':
        log=RUN/'prelaunch'/(args.tag+'.log');control=RUN/'prelaunch'/('worker-'+args.tag+'.json')
        assert not log.exists() and not control.exists()
        with log.open('wb') as out:
            child=subprocess.Popen([sys.executable,'-X','utf8',str(__file__),'worker','--tag',args.tag],stdin=subprocess.DEVNULL,
                stdout=out,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW|subprocess.DETACHED_PROCESS)
        print(json.dumps({'pid':child.pid,'tag':args.tag,'maximum_seconds':cfg['maximum_seconds']}))
    else:
        control=RUN/'prelaunch'/('worker-'+args.tag+'.json');start=time.monotonic()
        completed=[];result={'status':'running','tag':args.tag,'completed':completed}
        commands=[('operators',['06_component_operators.py','--attempt',args.tag+'-operators'])]
        commands += [(cid,['07_component_screen.py','--condition',cid,'--attempt',args.tag+'-'+cid]) for cid in cfg['conditions']]
        try:
            for name,argv in commands:
                left=int(cfg['maximum_seconds']-(time.monotonic()-start));assert left>0
                result.update(condition=name,elapsed_seconds=time.monotonic()-start);write(control,result)
                command='source /opt/sparsity-gpu/activate.sh; cd '+shlex.quote(LINUX)+'; exec timeout --signal=TERM --kill-after=10s '+str(left)+'s '+shlex.join([PYTHON,'-u',*argv])
                leaf=subprocess.run(['wsl.exe','-d','SparsityGPU','--exec','bash','-c',command])
                assert leaf.returncode==0,(name,leaf.returncode)
                completed.append(name)
            result['status']='complete'
        except Exception as exc:result.update(status='failed',error=str(exc))
        finally:result['elapsed_seconds']=time.monotonic()-start;write(control,result)

if __name__=='__main__':main()

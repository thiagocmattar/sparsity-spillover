"""Seal source bytes and only the four approved checkpoints/cache."""
import io
import json
import shutil
import tarfile
from runio import ROOT, REPO, read, write, record, verify, fs

def main():
    dest=ROOT/'bundles/input-001.tar.gz';dest.parent.mkdir(exist_ok=True)
    if dest.exists():raise ValueError('Fresh bundle required')
    entries={};reuse=[];checkpoints=[]
    def add(path,name,expected=None):
        row=record(path,REPO)
        if expected and (row['bytes'],row['sha256'])!=(expected['bytes'],expected['sha256']):raise ValueError(str(path))
        if name in entries and entries[name]!=path:raise ValueError('Duplicate '+name)
        entries[name]=path;reuse.append({'source':row,'copy':{'path':name,'bytes':row['bytes'],'sha256':row['sha256']}})
    for run in ['045-2026-09-20-pythia70m-kernel-grid','049-2026-09-22-pythia70m-short-row-limits']:
        source=REPO/'runs'/run
        rows=[r['copy'] for r in read(source/'provenance/reuse.json')['files'] if not r['copy']['path'].startswith(('inputs/','provenance/'))]
        rows+=read(source/'provenance/source-freeze.json')['files']
        for row in rows:add(source/row['path'],'dependencies/repository/runs/'+run+'/'+row['path'],row)
    run='051-2026-09-22-pythia70m-frozen-kernel-grid';source=REPO/'runs'/run
    for row in read(source/'provenance/source-freeze.json')['files']:add(source/row['path'],'dependencies/repository/runs/'+run+'/'+row['path'],row)
    add(source/'provenance/selection-final.json','dependencies/repository/runs/'+run+'/provenance/selection-final.json')
    run='054-2026-09-24-pythia31m-t2-ph';source=REPO/'runs'/run/'latency'
    add(source/'port.py','dependencies/repository/runs/'+run+'/latency/port.py')
    for name,row in read(source/'port-provenance.json')['files'].items():
        path=source/'kernel'/name
        if record(path,REPO)['sha256']!=row['port_sha256']:raise ValueError(name)
        add(path,'dependencies/repository/runs/'+run+'/latency/kernel/'+name)
    attempts={'31m':('054-2026-09-24-pythia31m-t2-ph',{'0':'002-20260924-131935-52b2ec52','0p1':'005-20260924-134241-3b66fc99'}),
              '70m':('043-2026-09-20-pythia70m-hz-h-only-ol1',{'0':'001-20260920-151358-cf1461a2','0p1':'004-20260920-151356-53f6e290'})}
    for size,(run,pairs) in attempts.items():
        for pair,attempt in pairs.items():
            source=REPO/'runs'/run/'artifacts/attempts'/attempt/'checkpoints/step_000712'
            name=f'inputs/checkpoints/{size}-k{pair}'
            for p in source.iterdir():
                if p.name in ['model.safetensors','config.json','checkpoint_metadata.json','generation_config.json']:add(p,name+'/'+p.name)
            checkpoints.append({'size':size,'pair':pair,'source_run':run,'source_attempt':attempt,'checkpoint':name,
                'files':[r['copy'] for r in reuse if r['copy']['path'].startswith(name+'/')]})
            if {r['path'].rsplit('/',1)[-1] for r in checkpoints[-1]['files']}!={'model.safetensors','config.json','checkpoint_metadata.json','generation_config.json'}:raise ValueError('Incomplete checkpoint')
    prior=REPO/'runs/063-2026-10-07-pythia14m-tile-specialized-kernel'
    cache=read(prior/'provenance/inputs.json')['validation'];add(prior/cache['path'],'inputs/validation.int32.bin',cache)
    add(prior/'provenance/pip-freeze.txt','provenance/pip-freeze.txt')
    write(ROOT/'provenance/inputs.json',{'checkpoints':checkpoints,'validation':next(r['copy'] for r in reuse if r['copy']['path']=='inputs/validation.int32.bin')})
    write(ROOT/'provenance/reuse.json',{'files':reuse})
    for p in sorted(ROOT.glob('*.py'))+sorted(ROOT.glob('*.sh'))+[ROOT/'config.json']+list((ROOT/'provenance').glob('*.json')):
        entries[p.relative_to(ROOT).as_posix()]=p
    rows=[]
    with tarfile.open(dest,'w:gz',compresslevel=3) as out:
        for name,path in sorted(entries.items()):
            row=record(path,REPO);rows.append({'path':name,'bytes':row['bytes'],'sha256':row['sha256']})
            info=tarfile.TarInfo(name);info.size=row['bytes']
            with fs(path).open('rb') as f:out.addfile(info,f)
        data=json.dumps({'files':rows},indent=2).encode();info=tarfile.TarInfo('prelaunch/bundle-inventory.json');info.size=len(data);out.addfile(info,io.BytesIO(data))
    write(ROOT/'prelaunch/bundle-receipt.json',{**record(dest,ROOT),'files':len(rows),'payload_bytes':sum(r['bytes'] for r in rows)})
    print(json.dumps(read(ROOT/'prelaunch/bundle-receipt.json')))

if __name__=='__main__':main()

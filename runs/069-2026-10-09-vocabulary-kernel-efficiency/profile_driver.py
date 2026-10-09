"""Bounded profiler invocations; permission failures remain explicit missing evidence."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from common import RUN, config, read, write, emit


def main():
    p=argparse.ArgumentParser();p.add_argument('--development',action='store_true');args=p.parse_args()
    cfg=config();root=RUN/('prelaunch/development-profiles' if args.development else 'artifacts/profiles')
    root.mkdir(parents=True,exist_ok=False)
    ncu=os.environ.get('NCU') or shutil.which('ncu');reports=[]
    if not ncu:
        write(root/'manifest.json',{'status':'unavailable','reason':'ncu executable missing','reports':[]});return
    version=subprocess.run([ncu,'--version'],capture_output=True,text=True,check=False)
    (root/'version.txt').write_text(version.stdout+version.stderr,encoding='utf-8')
    screen=RUN/('prelaunch/development-screen' if args.development else 'artifacts/screen')
    unavailable=False
    for folder in sorted(screen.iterdir()):
        result=read(folder/'result.json');names=list(dict.fromkeys(['native','legacy_n256']+result['shortlist']))
        indices=[0] if args.development else cfg['profiler_inputs']
        for name in names:
            for skip in ([False] if name=='native' else [False,True]):
                for index in indices:
                    stem=f'{folder.name}-{name}-'+('sparse' if skip else 'no-skip')+f'-i{index}'
                    cmd=[ncu,'--nvtx','--nvtx-include','head-target/','--target-processes','all',
                         '--section','LaunchStats','--section','SpeedOfLight','--section','Occupancy',
                         '--section','MemoryWorkloadAnalysis','--section','SchedulerStats','--section','WarpStateStats',
                         '--section','SourceCounters','--force-overwrite','--export',str(root/stem),
                         sys.executable,str(RUN/'06_profile.py'),'--model',folder.name,'--variant',name,'--input',str(index)]
                    if skip:cmd.append('--skip')
                    if args.development:cmd.append('--development')
                    emit(root,'profile',model=folder.name,variant=name,skip=skip,input=index)
                    try:
                        process=subprocess.run(cmd,capture_output=True,text=True,timeout=180,check=False)
                        output=process.stdout+process.stderr;code=process.returncode
                    except subprocess.TimeoutExpired as exc:
                        output='Profiler invocation timed out after 180 seconds';code=124
                    (root/(stem+'.txt')).write_text(output,encoding='utf-8')
                    identities=[json.loads(line.removeprefix('RUN069_IDENTITY ')) for line in output.splitlines()
                                if line.startswith('RUN069_IDENTITY ')]
                    reports.append({'name':stem,'returncode':code,'command':cmd,'report_exists':(root/(stem+'.ncu-rep')).exists(),
                                    'runtime_identity':identities[0] if len(identities)==1 else None})
                    if 'ERR_NVGPUCTRPERM' in output or 'driver resource' in output.lower():
                        unavailable=True;break
                if unavailable:break
            if unavailable:break
        if unavailable:break
    complete=bool(reports) and all(r['returncode']==0 and r['report_exists'] and r['runtime_identity'] for r in reports)
    write(root/'manifest.json',{'status':'unavailable' if unavailable else ('complete' if complete else 'partial'),'reports':reports,
         'interpretation':'Profiler replay/cache policy changes timing; use independent paired samples for latency. Unavailable counters do not establish a bottleneck.'})


if __name__=='__main__':main()

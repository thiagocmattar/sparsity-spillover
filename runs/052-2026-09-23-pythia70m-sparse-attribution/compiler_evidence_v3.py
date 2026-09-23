"""Retain the exact CUDA extension plus resource and PTX reports."""
from pathlib import Path
import shutil,subprocess
from support import RUN,sha,write


def retain(extension,dest):
    dest=Path(dest);dest.mkdir(parents=True,exist_ok=True)
    binary=Path(extension.__file__);target=dest/binary.name
    shutil.copyfile(binary,target)
    result={'binary':target.name,'sha256':sha(target),'source_sha256':sha(RUN/'followup_sparse.cu'),'reports':{}}
    for flag,label in [('--dump-resource-usage','resources'),('--dump-ptx','ptx'),('--dump-sass','sass')]:
        if shutil.which('cuobjdump') is None:
            result['reports'][label]={'error':'cuobjdump unavailable'};continue
        report=subprocess.run(['cuobjdump',flag,str(binary)],capture_output=True,text=True)
        path=dest/(binary.name+'.'+label+'.txt')
        path.write_text(report.stdout+'\n'+report.stderr,encoding='utf-8')
        result['reports'][label]={'path':path.name,'returncode':report.returncode,'sha256':sha(path)}
    write(dest/'manifest.json',result)
    return result

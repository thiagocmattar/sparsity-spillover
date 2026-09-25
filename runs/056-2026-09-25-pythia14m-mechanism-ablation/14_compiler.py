"""Retain actual JIT binary identities and resource reports outside timing."""
import os
from pathlib import Path
import subprocess
from io_utils import RUN, record, sha, write
from site_controls import base_joint, extension
from controls import mask


def main():
    destination=RUN/'runtime/compiler'
    destination.mkdir(parents=True,exist_ok=True)
    rows=[]
    for mode in ['frozen','t00','t10','t01','t11']:
        module=base_joint.extension() if mode=='frozen' else extension(**mask(mode))
        binary=Path(module.__file__)
        result=subprocess.run(['cuobjdump','--dump-resource-usage',str(binary)],
                              capture_output=True,text=True,check=True)
        output=destination/(mode+'.txt')
        output.write_text(result.stdout+result.stderr,encoding='utf-8')
        rows.append({'mode':mode,'binary_name':binary.name,'binary_bytes':binary.stat().st_size,
                     'binary_sha256':sha(binary),'resource_report':record(output)})
    write(destination/'manifest.json',{'binaries':rows,'extensions_directory':os.environ.get('TORCH_EXTENSIONS_DIR')})


if __name__=='__main__':main()

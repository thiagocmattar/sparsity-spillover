"""Bundle the actual repository commit; datasets/random replay travel separately."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from run_config import RUN_DIR,REPO_ROOT,write_json

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--tag',default='001')
    args=parser.parse_args()
    if not args.tag.isalnum():raise ValueError('Alphanumeric tag required')
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO_ROOT,text=True).strip()
    bundle=RUN_DIR/'prelaunch'/f'source-{args.tag}.bundle'
    if bundle.exists():raise FileExistsError(bundle)
    subprocess.run(['git','bundle','create',str(bundle),'HEAD'],cwd=REPO_ROOT,check=True)
    with bundle.open('rb') as handle:sha=hashlib.file_digest(handle,'sha256').hexdigest()
    receipt={'origin_commit':commit,'deployment_commit':commit,'path':bundle.relative_to(RUN_DIR).as_posix(),
             'bytes':bundle.stat().st_size,'sha256':sha}
    write_json(RUN_DIR/'prelaunch'/f'source-{args.tag}.receipt.json',receipt)
    print(json.dumps(receipt))

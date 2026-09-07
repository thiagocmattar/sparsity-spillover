"""Retrieve exact upstream Git blobs for a source-level compatibility audit only."""
import hashlib
import json
from urllib.request import urlopen, Request
from io_utils import RUN, read, write, record, fs, archived_run, sha

COMMIT='661f1fc841ed84d92f3fca5e5a94cffc5bf00ee5'
REPO='SakanaAI/sparser-faster-llms'
PATHS=['LICENSE','README.md','benchmark_inference.py','benchmark_base.py',
       'custom_models/twell_modules/matmul_d2t.cu','custom_models/twell_modules/matmul_t2d.cu',
       'custom_models/twell_modules/twell.py','custom_models/sparse_testing_utils.py']


def fetch(url):
    with urlopen(Request(url,headers={'User-Agent':'Run029-reproducibility-audit'}),timeout=60) as response:
        return response.read()


def main():
    tree=json.loads(fetch(f'https://api.github.com/repos/{REPO}/git/trees/{COMMIT}?recursive=1'))
    files=[]
    for path in PATHS:
        blob=next(r for r in tree['tree'] if r['path']==path)
        url=f'https://raw.githubusercontent.com/{REPO}/{COMMIT}/{path}'
        data=fetch(url)
        actual=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
        if actual!=blob['sha']:raise ValueError('Upstream Git blob identity mismatch')
        dest=RUN/'upstream-reference'/path;fs(dest.parent).mkdir(parents=True,exist_ok=True)
        if fs(dest).exists() and fs(dest).read_bytes()!=data:raise ValueError('Refuse changed upstream snapshot')
        if not fs(dest).exists():fs(dest).write_bytes(data)
        files.append({**record(dest),'git_blob_sha1':actual,'url':url})
    if sha(RUN/'upstream-reference/custom_models/twell_modules/matmul_t2d.cu')!=sha(archived_run(25)/'upstream/matmul_t2d.cu'):
        raise ValueError('Historical upstream reference differs')
    write(RUN/'provenance/sakana-upstream.json',{'commit':COMMIT,'files':files,
          'evaluation':'Source audit only; no unchanged full-model Pythia14M execution claimed.',
          'matched_comparator':'Frozen Run025 P0 signed-exact, shape-adapted descendant.'})
    print({'verified_git_blobs':len(files),'historical_t2d_identical':True})


if __name__=='__main__':main()

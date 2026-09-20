"""Assign adjacent scalar-path output columns to adjacent CUDA lanes."""
import subprocess
import sys
from io_utils import RUN,read,write,record


def main():
    for identifier,origin in [('opt019','opt001'),('opt020','opt015'),('opt021','opt016')]:
        source=RUN/'candidates'/origin;folder=RUN/'candidates'/identifier
        folder.mkdir(exist_ok=False);spec=read(source/'spec.json')
        cu=(source/'joint.cu').read_text()
        before='int col=blockIdx.y*256+lane*8+e,slot=local*256+lane*8+e;'
        after='int col=blockIdx.y*256+e*32+lane,slot=local*256+e*32+lane;'
        assert cu.count(before)==1
        (folder/'joint.cu').write_text(cu.replace(before,after),newline='\n')
        (folder/'joint.py').write_text((source/'joint.py').read_text().replace(origin,identifier),newline='\n')
        candidate=(source/'candidate.py').read_text().replace(origin,identifier)
        candidate=candidate.replace("'sparse_sites':['h','z']", "'scalar_column_mapping':'Adjacent lanes write adjacent columns','sparse_sites':['h','z']")
        (folder/'candidate.py').write_text(candidate,newline='\n')
        spec.update(kind='coalesced-scalar-columns',short_limit=spec.get('short_limit',2),
                    origin_manifest=record(source/'manifest.json'),
                    hypothesis='Contiguous addresses across lanes reduce memory transactions; per-output arithmetic unchanged from parent.')
        write(folder/'spec.json',spec)
        subprocess.run([sys.executable,str(RUN/'11_candidate.py'),'register',identifier],check=True)

if __name__=='__main__':main()

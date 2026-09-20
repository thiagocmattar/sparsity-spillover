"""Finer query scheduling and reversed causal-block order; dense attention only."""
import subprocess
import sys
from io_utils import RUN,read,write,record


def main():
    for identifier,parent,m,n,warps,reverse in [
            ('opt022','opt007',32,64,2,False),
            ('opt023','opt008',32,128,2,False),
            ('opt024','opt008',64,128,4,True)]:
        source=RUN/'candidates'/parent;folder=RUN/'candidates'/identifier
        old=read(source/'spec.json');attention=folder/'attention'
        attention.mkdir(parents=True,exist_ok=False)
        for path in (source/'attention').iterdir():
            if not path.is_file():continue
            text=path.read_text().replace(parent,identifier)
            if path.name=='kernel.cu':
                before=f"<64,{old['M']},{old['N']},{old['warps']},false,false,cutlass::bfloat16_t>"
                assert text.count(before)==1
                text=text.replace(before,f'<64,{m},{n},{warps},false,false,cutlass::bfloat16_t>')
                text=text.replace(f"dim3({2048//old['M']},1,8),{old['warps']*32},smem",f'dim3({2048//m},1,8),{warps*32},smem')
            if path.name=='flash_fwd_kernel.h':
                text=text.replace(f"((bidh*{2048//old['M']}+m_block)*{old['warps']}+tidx/32)*4",f'((bidh*{2048//m}+m_block)*{warps}+tidx/32)*4')
                if reverse:
                    before='inline __device__ void compute_attn(const Params &params) {\n    const int m_block = blockIdx.x;'
                    assert text.count(before)==1
                    text=text.replace(before,'inline __device__ void compute_attn(const Params &params) {\n    const int m_block = gridDim.x - 1 - blockIdx.x;')
            (attention/path.name).write_text(text,newline='\n')
        candidate=(source/'candidate.py').read_text().replace(parent,identifier)
        candidate=candidate.replace(f"'attention_tile':[{old['M']},{old['N']}]",f"'attention_tile':[{m},{n}]")
        candidate=candidate.replace(f"'attention_warps':{old['warps']}",f"'attention_warps':{warps},'reversed_query_schedule':{reverse}")
        (folder/'candidate.py').write_text(candidate,newline='\n')
        write(folder/'spec.json',{'kind':'dense-attention','M':m,'N':n,'warps':warps,
            'reversed_query_schedule':reverse,'origin_manifest':record(source/'manifest.json'),
            'hypothesis':'Smaller query groups provide more independent GPU blocks; long causal blocks first may improve scheduling balance.',
            'numerical_note':'Same gates and dense attention; original numerical bounds and causal checks.'})
        subprocess.run([sys.executable,str(RUN/'11_candidate.py'),'register',identifier],check=True)

if __name__=='__main__':main()

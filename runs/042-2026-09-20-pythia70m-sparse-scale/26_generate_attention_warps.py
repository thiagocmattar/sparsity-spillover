"""One remaining dense-attention schedule: M128/N128 with four warps."""
import subprocess
import sys
from io_utils import RUN,read,write,record


def main():
    parent,identifier='opt011','opt031'
    source=RUN/'candidates'/parent;folder=RUN/'candidates'/identifier
    attention=folder/'attention';attention.mkdir(parents=True,exist_ok=False)
    for path in (source/'attention').iterdir():
        if not path.is_file():continue
        text=path.read_text().replace(parent,identifier)
        if path.name=='kernel.cu':
            old='<64,128,128,8,false,false,cutlass::bfloat16_t>'
            assert text.count(old)==1
            text=text.replace(old,'<64,128,128,4,false,false,cutlass::bfloat16_t>')
            assert 'dim3(16,1,8),256,smem' in text
            text=text.replace('dim3(16,1,8),256,smem','dim3(16,1,8),128,smem')
        if path.name=='flash_fwd_kernel.h':
            text=text.replace('((bidh*16+m_block)*8+tidx/32)*4','((bidh*16+m_block)*4+tidx/32)*4')
        (attention/path.name).write_text(text,newline='\n')
    candidate=(source/'candidate.py').read_text().replace(parent,identifier)
    candidate=candidate.replace("'attention_warps':8","'attention_warps':4")
    (folder/'candidate.py').write_text(candidate,newline='\n')
    write(folder/'spec.json',{'kind':'dense-attention','M':128,'N':128,'warps':4,
        'origin_manifest':record(source/'manifest.json'),
        'hypothesis':'Keep the numerically qualified key-block reduction order while assigning more query rows to each warp; tests computation efficiency versus register pressure.',
        'numerical_note':'No change in gates, operands, softmax or original numerical bounds.'})
    subprocess.run([sys.executable,str(RUN/'11_candidate.py'),'register',identifier],check=True)


if __name__=='__main__':main()

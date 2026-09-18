"""Materialize the narrowly scoped 70M changes from retained K050 components.

Run before execution only. Original archived kernels remain byte-identical.
The generated CUDA/Python files, source hashes and this transformation are kept.
"""
from pathlib import Path
import hashlib
import json

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
BASE=ROOT/'runs/028-2026-09-06-pythia14m-all-site-sparse-kernels'
records=[]

def source(rel):
    p=BASE/rel;raw=p.read_bytes()
    records.append({'path':p.relative_to(ROOT).as_posix(),'sha256':hashlib.sha256(raw).hexdigest()})
    return raw.decode()

def write(rel,text):
    p=HERE/'kernel'/rel;p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(text,encoding='utf-8',newline='\n')

def replace(text,old,new):
    assert old in text,old
    return text.replace(old,new)

def main():
    assert not (HERE/'artifacts/attempts').exists(), 'Never rewrite an executed port'
    t=source('candidates/k050/candidate.py').replace('run028_k050_','run035_norm_').replace('128','512')
    write('norm/candidate.py',t)
    write('norm/LICENSE-PYTORCH',source('candidates/k050/LICENSE-PYTORCH'))
    t=source('candidates/k050/norm.cu')
    t=replace(t,'one warp per width128 row, four rows/CTA','four warps per width512 row, one row/CTA')
    t=replace(t,'int lane=threadIdx.x&31,row=blockIdx.x*4+threadIdx.x/32;',
        'int lane=threadIdx.x&31,warp=threadIdx.x/32,row=blockIdx.x;\n    __shared__ Moments partial[4];\n    __shared__ float mean_shared,var_shared;')
    t=replace(t,'x+row*128)[lane]','x+row*512)[threadIdx.x]')
    t=replace(t,'float mean=__shfl_sync(0xffffffff,state.mean,0);\n    float variance=__shfl_sync(0xffffffff,state.m2,0)/128.f;',
        '''if(lane==0)partial[warp]=state;
    __syncthreads();
    // Same four-warp reduction tree as the pinned native vectorized LayerNorm.
    for(int offset=2;offset>0;offset/=2){
        if(lane==0 && warp<offset)partial[warp]=merge(partial[warp],partial[warp+offset]);
        __syncthreads();
    }
    if(threadIdx.x==0){mean_shared=partial[0].mean;var_shared=partial[0].m2/512.f;}
    __syncthreads();
    float mean=mean_shared,variance=var_shared;''')
    for name in ['wa','ba','wm','bm']:
        t=replace(t,f'({name})[lane]',f'({name})[threadIdx.x]')
    for name in ['a','m']:
        t=replace(t,f'({name}+row*128)[lane]',f'({name}+row*512)[threadIdx.x]')
    t=t.replace('==128','==512').replace('width128','width512').replace('x.numel()/128','x.numel()/512')
    t=replace(t,'<<<(rows+3)/4,128,0,stream>>>','<<<rows,128,0,stream>>>')
    write('norm/norm.cu',t)

    t=source('candidates/k042/candidate.py').replace('run028_k042_','run035_projection_').replace('reshape(-1,128)','reshape(-1,512)')
    write('projection/candidate.py',t)
    t=source('candidates/k042/projection.cu').replace('==128','==512').replace('Mx128','Mx512').replace('(n==384 || n==512)','(n==1536 || n==2048)')
    write('projection/projection.cu',t)

    t=source('candidates/k049/candidate.py').replace('run028_k049_','run035_joint_')
    t=t.replace('h.reshape(-1,512)','h.reshape(-1,2048)').replace('reshape(-1,128)','reshape(-1,512)')
    t=t.replace('(r.shape[0]//8,8,6)','(r.shape[0]//8,4,8,6)')
    write('joint/candidate.py',t)
    t=source('candidates/k049/joint.cu')
    t=t.replace('unsigned tiles=0;','unsigned tiles[4]={};')
    t=t.replace('out.tiles|=1u<<(base/16);','out.tiles[base/512]|=1u<<((base/16)%32);')
    t=t.replace('out.tiles|=1u<<(base/16+1);','out.tiles[base/512]|=1u<<((base/16+1)%32);')
    t=t.replace('a.i0*128+col','a.i0*512+col').replace('a.i1*128+col','a.i1*512+col')
    t=t.replace('unsigned active_tiles','const unsigned* active_tiles')
    t=replace(t,'unsigned pending=Skip?active_tiles:(K==512?0xffffffffu:0xffu);',
        'for(int word=0;word<K/512;++word){\n    unsigned pending=Skip?active_tiles[word]:0xffffffffu;')
    t=t.replace('(K/16-__popc(pending))*4','(32-__popc(pending))*4')
    t=t.replace('int base=(__ffs(pending)-1)*16','int base=word*512+(__ffs(pending)-1)*16')
    t=replace(t,'if constexpr(Count)issued+=4;\n        }\n    }\n}',
        'if constexpr(Count)issued+=4;\n        }\n    }\n    }\n}')
    t=t.replace('(blockIdx.x*8+warp)*6','((blockIdx.x*4+blockIdx.y)*8+warp)*6')
    t=t.replace('unsigned h_masks[8],z_masks[8];','unsigned h_masks[8][4],z_masks[8][4];')
    t=t.replace('unsigned h_tiles=0xffffffffu,z_tiles=0xffu;',
        'unsigned h_tiles[4]={~0u,~0u,~0u,~0u},z_tiles[4]={~0u,0,0,0};')
    t=t.replace('inspect<512>(h','inspect<2048>(h').replace('inspect<128>(z','inspect<512>(z')
    t=replace(t,'if(lane==0){h_masks[local]=h_complex?ha.tiles:0;z_masks[local]=z_complex?za.tiles:0;}',
        'if(lane==0)for(int w=0;w<4;++w){h_masks[local][w]=h_complex?ha.tiles[w]:0;z_masks[local][w]=z_complex?za.tiles[w]:0;}')
    t=t.replace('int col=lane*4+e,slot=', 'int col=blockIdx.y*128+lane*4+e,slot=')
    t=t.replace('out[row*128+col]','out[row*512+col]').replace('residual[row*128+col]','residual[row*512+col]')
    t=replace(t,'h_tiles=0;z_tiles=0;for(int i=0;i<8;++i){h_tiles|=h_masks[i];z_tiles|=z_masks[i];}',
        'unsigned any=0;for(int w=0;w<4;++w){h_tiles[w]=z_tiles[w]=0;for(int i=0;i<8;++i){h_tiles[w]|=h_masks[i][w];z_tiles[w]|=z_masks[i][w];}any|=h_tiles[w]|z_tiles[w];}')
    t=t.replace('if((h_tiles|z_tiles)==0)','if(any==0)').replace('warp<4?128:0','warp<4?512:0').replace('warp<4?32:0','warp<4?128:0')
    t=t.replace('col=warp*32;','col=blockIdx.y*128+warp*32;')
    t=t.replace('accumulate<512,Skip,Count>(h','accumulate<2048,Skip,Count>(h').replace('accumulate<128,Skip,Count>(z','accumulate<512,Skip,Count>(z')
    t=t.replace('slot=lr*128+c;','slot=lr*128+c%128;')
    t=t.replace('out[r*128+c]','out[r*512+c]').replace('residual[r*128+c]','residual[r*512+c]')
    t=t.replace('h.size(1)==512','h.size(1)==2048').replace('{m,128}','{m,512}')
    t=t.replace('{128,512}','{512,2048}').replace('{128,128}','{512,512}').replace('{512,128}','{2048,512}')
    t=t.replace('numel()==128','numel()==512').replace('(m/8)*8*6','(m/8)*4*8*6')
    t=t.replace('dim3(m/8),256','dim3(m/8,4),256').replace('M8 H512 Z128','M8 H2048 Z512')
    write('joint/joint.cu',t)

    for name in ['flash_fwd_kernel.h','sparse_gemm.h','LICENSE-FLASH']:
        write('attention/'+name,source('candidates/k035/'+name))
    t=source('candidates/k035/candidate.py').replace('run028_k035_','run035_attention_')
    for old,new in [('(1,4,2048,32)','(1,8,2048,64)'),('B1 H4 T2048 D32','B1 H8 T2048 D64'),
                    ('(4,2048)','(8,2048)'),('(2,4,2048)','(2,8,2048)'),('(2,4,2048,32)','(2,8,2048,64)'),
                    ('(4,32,2,4,4)','(8,32,2,4,4)'),('(2,4,1024,32)','(2,8,1024,64)'),
                    ('(2,4,16,32)','(2,8,16,32)'),('(4,32,2,4,3)','(8,32,2,4,3)')]:t=t.replace(old,new)
    write('attention/candidate.py',t)
    # Diagnostics keep independent integer conservation at the new dimensions.
    t=source('115_hybrid_diagnostics.py')
    t=t.replace('def hybrid_counts(value,fast_weights=True,skip=True):','def hybrid_counts(value,fast_weights=True,skip=True):\n    n=512')
    t=t.replace('(m//8)*(k//16)*16','(m//8)*(k//16)*(n//8)').replace('int(active.sum())*16','int(active.sum())*(n//8)').replace('int(nnz[simple].sum())*128','int(nnz[simple].sum())*n')
    t=t.replace('!=131072','!=2097152').replace('!=32768','!=524288').replace('!=147456','!=589824')
    t=t.replace("('qkv_projection',384)","('qkv_projection',1536)").replace("('mlp_w1',512)","('mlp_w1',2048)").replace("('mlp_w2',128)","('mlp_w2',512)").replace("('attention_output_projection',128)","('attention_output_projection',512)")
    (HERE/'diagnostics.py').write_text(t,encoding='utf-8',newline='\n')
    (HERE/'provenance').mkdir(exist_ok=True)
    (HERE/'provenance/port-origin.json').write_text(json.dumps({'sources':records,'identity':'k050-70m-v1'},indent=2)+'\n')

if __name__=='__main__':main()

"""Correct the port to the observed 70M native attention reduction schedule.

This is a numerical compatibility correction, not a tile-performance search.
Native profiler evidence: D64/M128/N128/4warps, grid16x1x8, no KV splitting.
Run after00_port_sources.py, before freezing the final scientific cohort.
"""
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent

def main():
    assert not (HERE/'results/70m-final-kernel.json').exists()
    p=HERE/'kernel/attention/flash_fwd_kernel.h';t=p.read_text()
    split=t.index('template<typename Kernel_traits, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_softcap, bool Split, bool Append_KV, typename Params>')
    a,b=t[:split],t[split:]
    assert 'run035_q_issued' not in a, 'Correction already applied'
    a=a.replace('const int tidx = threadIdx.x;','''const int tidx = threadIdx.x;
    int run035_q_issued=0,run035_q_skipped=0,run035_p_issued=0,run035_p_skipped=0;
    const int run035_offset=((bidh*16+m_block)*4+tidx/32)*4;
    if constexpr(Kernel_traits::Run028Count) if((tidx&31)==0){
        for(int c=0;c<4;++c)params.run028_stats[run035_offset+c]=0;
    }''',1)
    old='FLASH_NAMESPACE::gemm</*A_in_regs=*/Kernel_traits::Is_Q_in_regs>('
    assert a.count(old)==2
    a=a.replace(old,'FLASH_NAMESPACE::run028_gemm<Kernel_traits::Run028Skip,Kernel_traits::Run028Count,Kernel_traits::Is_Q_in_regs>(')
    old='smem_thr_copy_Q, smem_thr_copy_K\n        );'
    assert a.count(old)==2
    a=a.replace(old,'smem_thr_copy_Q, smem_thr_copy_K, run035_q_issued, run035_q_skipped\n        );')
    old='FLASH_NAMESPACE::gemm_rs(acc_o, tOrP, tOrVt, tOsVt, tiled_mma, smem_tiled_copy_V, smem_thr_copy_V);'
    assert a.count(old)==2
    a=a.replace(old,'FLASH_NAMESPACE::run028_gemm_rs<Kernel_traits::Run028Skip,Kernel_traits::Run028Count>(acc_o,tOrP,tOrVt,tOsVt,tiled_mma,smem_tiled_copy_V,smem_thr_copy_V,run035_p_issued,run035_p_skipped);')
    index=a.rfind('\n}')
    a=a[:index]+'''
    if constexpr(Kernel_traits::Run028Count) if((tidx&31)==0){
        params.run028_stats[run035_offset]=run035_q_issued;
        params.run028_stats[run035_offset+1]=run035_q_skipped;
        params.run028_stats[run035_offset+2]=run035_p_issued;
        params.run028_stats[run035_offset+3]=run035_p_skipped;
    }
'''+a[index:]
    p.write_text(a+b,newline='\n')
    p=HERE/'kernel/attention/kernel.cu';t=p.read_text()
    # Avoid the pybind/cute overload ambiguity found during v1 compilation.
    t=t.replace('void forward(', 'void run035_forward(').replace('&forward);','&run035_forward);')
    t=t.replace('Frozen no-prefix policy; two KV splits.','Native128x128 causal schedule, no KV splitting; exact-zero MMA bypass.')
    t=t.replace('Flash_fwd_kernel_traits<64,64,256,4','Flash_fwd_kernel_traits<64,128,128,4')
    t=t.replace('compute_attn_splitkv<Traits<Skip,Count>,true,false,false,true,true,false,true,false>(p);',
        'compute_attn<Traits<Skip,Count>,false,true,false,false,true,true,false,false>(p);')
    start=t.index('__global__ void combine(');end=t.index('template<bool Skip,bool Count> void launch',start)
    t=t[:start]+t[end:]
    t=t.replace('dim3(32,2,8)','dim3(16,1,8)').replace('8*32*2*4*4','8*16*1*4*4').replace('8*32*2*4*3','8*16*1*4*3')
    t=t.replace('p.num_splits=2','p.num_splits=1')
    t=t.replace('    run035_flash::combine<<<1024,128,0,stream>>>(p);\n    C10_CUDA_KERNEL_LAUNCH_CHECK();\n','')
    p.write_text(t,newline='\n')
    p=HERE/'kernel/attention/candidate.py';t=p.read_text().replace('(8,32,2,4,4)','(8,16,1,4,4)').replace('(8,32,2,4,3)','(8,16,1,4,3)');p.write_text(t,newline='\n')
    for name in ['diagnostics.py','10_test_operators.py']:
        p=HERE/name;p.write_text(p.read_text().replace('589824','557056'),newline='\n')
    for name in ['config.json','provenance/candidates.json','replay.py','kernel/candidate.py','03_execute.py','09_verify_retrieval.py','test_contract.py']:
        p=HERE/name;t=p.read_text().replace('k050-70m-v1','k050-70m-v2')
        t=t.replace('K035-derived H8 D64, two KV splits, no prefix shortcut','K050 exact-zero MMA bypass in native H8 D64 M128N128 unsplit Flash schedule; no prefix shortcut')
        p.write_text(t,newline='\n')
    p=HERE/'provenance/port-origin.json';v=json.loads(p.read_text())
    v.update(identity='k050-70m-v2',attention_correction='Observed pinned70M native128x128 unsplit schedule; preserve original MMA skipping and softmax operation order; not a performance search')
    p.write_text(json.dumps(v,indent=2)+'\n')

if __name__=='__main__':main()

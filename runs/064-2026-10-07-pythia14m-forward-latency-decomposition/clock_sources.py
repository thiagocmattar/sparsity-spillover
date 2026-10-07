"""Generate diagnostic-only timestamp copies from the sealed production bytes."""
from pathlib import Path

HELPER='''
__device__ __forceinline__ unsigned long long tick(){
    unsigned long long value; asm volatile("mov.u64 %0, %%clock64;" : "=l"(value) :: "memory"); return value;
}
__device__ __forceinline__ void save_ticks(long long* ticks,int warp,int lane,unsigned long long a,unsigned long long b,unsigned long long c,unsigned long long d,unsigned long long e){
    if(lane==0 && warp<4){int slot=(blockIdx.x*4+warp)*5;ticks[slot]=a;ticks[slot+1]=b;ticks[slot+2]=c;ticks[slot+3]=d;ticks[slot+4]=e;}
}
'''

def once(text, old, new):
    if text.count(old)!=1:raise ValueError('Timestamp patch anchor changed: '+old[:80])
    return text.replace(old,new)

def generate(source,specialized):
    text=Path(source).read_text()
    text=once(text,'using bf16=__nv_bfloat16;','using bf16=__nv_bfloat16;\n'+HELPER)
    signature='long long* stats,float th,float tz'+(')' if specialized else ',int gh,int gz,bool fast_weights)')
    text=once(text,signature,signature[:-1]+',long long* ticks)')
    marker='int lane=threadIdx.x&31,warp=threadIdx.x/32,first_row=blockIdx.x*8;'
    text=once(text,marker,marker+'\n    unsigned long long t0=tick();')
    marker='float ah[4][4]={},az[4][4]={};int hi=0,hb=0,zi=0,zb=0;'
    text=once(text,marker,marker+'\n    unsigned long long t1=tick();')
    hline=next(line for line in text.splitlines() if 'accumulate<512,Skip,Count>(' in line)
    zline=next(line for line in text.splitlines() if 'accumulate<128,Skip,Count>(' in line)
    text=once(text,hline,hline+'\n    unsigned long long t2=tick();')
    text=once(text,zline,zline+'\n    unsigned long long t3=tick();')
    if specialized:
        marker='    if constexpr(Count){if(lane==0){'
        text=once(text,marker,'    save_ticks(ticks,warp,lane,t0,t1,t2,t3,tick());\n'+marker)
    else:
        text=once(text,'if((h_tiles|z_tiles)==0){counters','if((h_tiles|z_tiles)==0){auto dead=tick();save_ticks(ticks,warp,lane,t0,dead,dead,dead,dead);counters')
        marker='    counters<Count>(stats,warp,lane,hi,hb,zi,zb,h_scalar,z_scalar);'
        text=once(text,marker,'    save_ticks(ticks,warp,lane,t0,t1,t2,t3,tick());\n'+marker)
    marker='double th,double tz,int gh,int gz,bool skip,bool fast_weights,bool count)'
    text=once(text,marker,marker[:-1]+',torch::Tensor ticks)')
    launch=next(line for line in text.splitlines() if '#define LAUNCH' in line)
    text=once(text,launch,launch[:-1]+',reinterpret_cast<long long*>(ticks.data_ptr()))')
    return text

if __name__=='__main__':
    import sys
    base=Path(sys.argv[1]);dest=Path(__file__).parent
    for specialized,name in [(False,'generic'),(True,'joint')]:
        (dest/('clocked_'+name+'.cu')).write_text(generate(base/(name+'.cu'),specialized),newline='\n')

"""Independent native-precision gate, projection, work-count and replay checks."""
import argparse
import time
import torch
from head import extension, pack_weight
from io_utils import RUN, configure_cuda, record, write


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--tag',required=True);args=parser.parse_args()
    output=RUN/f'prelaunch/component-{args.tag}.json'
    if output.exists():raise ValueError('A fresh component tag is required')
    started=time.monotonic();configure_cuda(2801);ext=extension();rows=[]
    with torch.inference_mode():
        for kappa in (0.,.1):
            for case in ('random','dead','signed','equality','underflow','all_live'):
                native=torch.randn(32,128,device='cuda')
                if case=='dead':native[::8,::16]=-1
                if case in ('signed','all_live'):native[::8,::16]=1.;native[:,1::16]=-3.
                if case=='equality':
                    native[::8,::16]=kappa
                    native[::8,0]=torch.nextafter(torch.tensor(kappa,device='cuda'),torch.tensor(float('inf'),device='cuda'))
                if case=='underflow':native[::8,::16]=kappa+1e-8 if kappa else 1e-12
                live=native[::8,::16]>kappa
                gated=native.reshape(4,8,8,16).masked_fill(~live[:,None,:,None],0).reshape(32,128)
                packed=torch.empty_like(gated,dtype=torch.float16);support=torch.empty(4,device='cuda',dtype=torch.int32)
                ext.prepare(gated,packed,support,kappa)
                expected_mask=(live.to(torch.int32)*(1<<torch.arange(8,device='cuda'))).sum(-1).int()
                if not torch.equal(packed,gated.half()) or not torch.equal(support,expected_mask):raise ValueError('Native certificate or cast differs')
                for m,n in ((8,128),(16,128),(16,256),(16,512)):
                    vocab=263  # exercises output-tail masks for each supported N tile
                    weight=(torch.randn(vocab,128,device='cuda')*.05).half()
                    mma_weight=pack_weight(weight)
                    reference=packed@weight.T
                    out=torch.empty_like(reference);control=torch.empty_like(reference)
                    stats=torch.empty((32//m,(vocab+n-1)//n,2),device='cuda',dtype=torch.int64)
                    ext.project(packed,support,mma_weight,out,stats,True,True,m,n)
                    unions=live.reshape(32//m,m//8,8).any(1)
                    potential=(32//m)*((vocab+n-1)//n)*8*(n//8)
                    issued=int(unions.sum())*((vocab+n-1)//n)*(n//8)
                    if stats.sum((0,1)).cpu().tolist()!=[issued,potential-issued]:raise ValueError('Independent issued/skipped MMA counts differ')
                    if not torch.allclose(out,reference,atol=.004,rtol=.004):raise ValueError('Independent matmul differs')
                    ext.project(packed,support,mma_weight,control,stats,False,True,m,n)
                    if not torch.equal(out,control) or stats.sum((0,1)).cpu().tolist()!=[potential,0]:raise ValueError('Skip/no-skip differs')
                    graph=torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        ext.prepare(gated,packed,support,kappa)
                        ext.project(packed,support,mma_weight,out,stats,True,False,m,n)
                    changed=native.neg();new_live=changed[::8,::16]>kappa
                    gated.copy_(changed.reshape(4,8,8,16).masked_fill(~new_live[:,None,:,None],0).reshape(32,128))
                    graph.replay()
                    if not torch.allclose(out,gated.half()@weight.T,atol=.004,rtol=.004):raise ValueError('Changed-input replay differs')
                    # Reset before testing the next consumer tile.
                    gated.copy_(native.reshape(4,8,8,16).masked_fill(~live[:,None,:,None],0).reshape(32,128));ext.prepare(gated,packed,support,kappa)
                    rows.append({'kappa':kappa,'case':case,'tile':[m,n],'pass':True,'issued':issued,'potential':potential,'replay':True})
    torch.cuda.synchronize()
    write(output,{'status':'passed','rows':rows,'seconds':time.monotonic()-started,
                  'gpu':torch.cuda.get_device_name(),'torch':torch.__version__,'cuda':torch.version.cuda,
                  'source':record(RUN/'head.cu'),'scope':'synthetic component qualification, not full-model qualification or accepted timing'})
    print({'status':'passed','checks':len(rows),'seconds':time.monotonic()-started},flush=True)


if __name__=='__main__':main()

"""One NVTX-delimited eager projection, separate from unprofiled latency evidence."""
import argparse
import torch
from common import RUN, config, read, verify, runtime, support
from kernel import make, dense


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--variant',required=True)
    p.add_argument('--skip',action='store_true');p.add_argument('--input',type=int,default=0);p.add_argument('--development',action='store_true')
    args=p.parse_args();cfg=config();runtime(cfg['runtime_seed'])
    root=RUN/('prelaunch/development-capture' if args.development else 'artifacts/capture')
    row=next(r for r in read(root/'manifest.json')['files'] if r['path'].endswith(args.model+'.pt'))
    data=torch.load(verify(row),map_location='cpu',weights_only=True)
    with torch.inference_mode():
        linear=torch.nn.Linear(128,50304,bias=False,device='cuda',dtype=torch.float32)
        linear.weight.data.copy_(data['weight']);pre=data['pre'][args.input].cuda();spec=data['gate']
        live=pre.reshape(-1,128)[::16,::16]>spec['kappa']
        x=(pre.reshape(128,16,8,16)*live[:,None,:,None]).reshape(2048,128).half()
        if args.variant=='native':
            head=dense(linear);operation=lambda:torch.nn.functional.linear(x,head.packed_weight)
        else:
            variant=next(v for v in cfg['variants'] if v['name']==args.variant)
            head=make(linear,spec,variant,args.skip);head(pre)
            head.x_half=x;head.support=support(pre,spec['kappa']);operation=head.project
        for _ in range(10):operation()
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_push('head-target');operation();torch.cuda.nvtx.range_pop()
        torch.cuda.synchronize()


if __name__=='__main__':main()

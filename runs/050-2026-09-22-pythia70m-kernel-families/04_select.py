"""Freeze shared site/layer policies from matched training component screens."""
import argparse,math
from support import RUN,read,write,sha

def select(screens, *, dense_ids=('dense_native','dense_fused','dense_dot')):
 keys=[f'{s}.{i}' for s in ('h','z') for i in range(6)]
 cells=[f'{c}:{split}:' for c in ('c24','c25') for split in ('development','confirmation')]
 dense={};details={};policies={f:{} for f in 'abcd'}
 for key in keys:
  # Dense controls occur in every independent screen. Normalize within each
  # screen; never compare absolute candidate times across sessions.
  def dense_score(name):
   rows=[s[c+key] for s in screens for c in cells]
   if not all(name in r and r[name]['qualified'] for r in rows):return math.inf
   return max(r[name]['host_ms']/r['dense_native']['host_ms'] for r in rows)
  choice=min(dense_ids,key=lambda n:(dense_score(n),n))
  assert math.isfinite(dense_score(choice))
  dense[key]=choice;detail={'dense':choice,'families':{}}
  for family in 'abcd':
   scores={}
   names={n for s in screens for n in s[cells[0]+key] if n.startswith(family+'_')}
   for name in names:
    ratios=[];valid=True
    for s in screens:
     if name not in s[cells[0]+key]:continue
     for c in cells:
      row=s[c+key];r=row.get(name)
      valid &= r is not None and r['qualified']
      if r is not None:ratios.append(r['host_ms']/row[choice]['host_ms'])
    if valid and len(ratios)>=4:scores[name]=max(ratios)
   winner=min(scores,key=lambda n:(scores[n],n)) if scores else None
   promote=winner is not None and scores[winner]<=.95
   policies[family][key]=winner if promote else choice
   detail['families'][family]={'scores':scores,'best':winner,'promoted':promote}
  details[key]=detail
 ranked=sorted((f for f in policies if policies[f]!=dense),
               key=lambda f:sum(math.log(details[k]['families'][f]['scores'].get(policies[f][k],1.)) for k in keys))[:2]
 ablations={}
 if 'c' in ranked:
  ablations['sparse_c_no_skip']={k:v+'_noskip' if v.startswith('c_') else v for k,v in policies['c'].items()}
 return {'dense':dense,'policies':{'sparse_'+f:policies[f] for f in ranked},'ablations':ablations,'details':details,
         'selection_data':'training blocks0:64 development,64:128 confirmation; both kappas',
         'criterion':'shared minimax ratio; sparse requires <=0.95 in every condition/split against matched dense'}

def main():
 p=argparse.ArgumentParser();p.add_argument('--screens',nargs='+',required=True);p.add_argument('--output',required=True);p.add_argument('--dense-ids',default='dense_native,dense_fused,dense_dot');a=p.parse_args()
 paths=[RUN/'artifacts'/s/'summary.json' for s in a.screens]
 result=select([read(p) for p in paths],dense_ids=tuple(a.dense_ids.split(',')))
 result['screen_sources']=[{'path':p.relative_to(RUN).as_posix(),'sha256':sha(p)} for p in paths]
 write(RUN/'provenance'/a.output,result)
 print({k:v for k,v in result.items() if k in ('dense','policies')})

if __name__=='__main__':main()

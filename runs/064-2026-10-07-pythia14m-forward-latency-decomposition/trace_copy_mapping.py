"""Post-hoc native mapping: retain graph copy nodes enclosed by one gate stage.

Raw failed strict mappings remain unchanged. This only annotates retained
activity durations; it never changes a measurement or an executed graph.
"""
import collections
import difflib

def map_native_copies(trace):
    if trace['mode']!='native' or trace['mapping']['mapped']:return trace['mapping']
    eager=trace['eager']['kernels'];graph=trace['graph']['kernels'];repeats=trace['graph']['repeats']
    if not graph or len(graph)%repeats:return {'mapped':False,'reason':'Incomplete graph trace'}
    count=len(graph)//repeats;first=graph[:count];labels=[];insertions=[]
    matcher=difflib.SequenceMatcher(a=[r['name'] for r in eager],b=[r['name'] for r in first],autojunk=False)
    for tag,i,j,a,b in matcher.get_opcodes():
        if tag=='equal':
            if any(not r['stage'] for r in eager[i:j]):return {'mapped':False,'reason':'Unattributed eager kernel'}
            labels.extend(r['stage'] for r in eager[i:j]);continue
        if tag!='insert' or not 0<i<len(eager):return {'mapped':False,'reason':'Non-copy kernel signature changed'}
        stage=eager[i-1]['stage']
        if stage!=eager[i]['stage'] or not stage or not stage.endswith(('.h_gate','.z_gate')):
            return {'mapped':False,'reason':'Copy not enclosed by one gate stage'}
        for row in first[a:b]:
            if row['name'] not in {'memcpy32_post','memcpy128'}:return {'mapped':False,'reason':'Unknown graph-only kernel'}
            labels.append(stage);insertions.append({'stage':stage,'name':row['name'],'graph_index':a})
    counts=collections.Counter(r['stage'] for r in insertions)
    if counts!=collections.Counter({f'layer{i}.{site}_gate':1 for i in range(6) for site in ['h','z']}):
        return {'mapped':False,'reason':'Expected exactly one graph copy per h/z gate per layer'}
    if len(labels)!=count or any(row['name']!=first[i%count]['name'] for i,row in enumerate(graph)):
        return {'mapped':False,'reason':'Replay signature changed'}
    return {'mapped':True,'kernels':[{**row,'stage':labels[i%count],'replay':i//count} for i,row in enumerate(graph)],
        'kernels_per_replay':count,'mapping':'All computational kernels match exact eager order; exactly12 graph-only memcpy activities are enclosed between kernels of the same h/z gate; all replay signatures match',
        'graph_only_gate_copies':insertions,'raw_strict_mapping_retained':True}

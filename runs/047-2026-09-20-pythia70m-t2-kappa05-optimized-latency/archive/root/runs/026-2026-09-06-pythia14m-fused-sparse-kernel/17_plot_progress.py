"""Plot recorded development progress and separately frozen final replication."""
from pathlib import Path
import csv
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RUN = Path(__file__).resolve().parent
rows = list(csv.DictReader((RUN/'autoresearch/progress.csv').open()))[:14]
summary = json.loads((RUN/'autoresearch/final-summary.json').read_text())
fig, (search, final) = plt.subplots(1, 2, figsize=(10.2, 4.4), gridspec_kw={'width_ratios':[1.55,1]})
for valid, color, marker, label in [(True,'#0072B2','o','Qualified'),(False,'#D55E00','x','Fails numerical gate')]:
    selected = [r for r in rows if (r['qualified']=='True') == valid]
    search.scatter([int(r['iteration'].split('-')[0]) for r in selected],
        [float(r['speedup_matched_dense']) for r in selected],c=color,marker=marker,label=label,zorder=3)
search.step(range(1,15),[float(r['best_so_far_matched_dense']) for r in rows],where='post',color='#0072B2',lw=1.5,label='Best qualified so far')
search.set(xlabel='Development iteration',ylabel='Paired geometric-mean speedup (x)',xlim=(.5,14.5),ylim=(0,4.5),title='Search history (16 inputs x 5 passes)')
search.set_xticks([1,3,5,7,9,11,14])
search.legend(loc='upper left',frameon=False,fontsize=8)
keys = ['joint_sparse_projections_only','rope_only_dense_projections','winner']
labels = ['Sparse joint\nW2/Wo only','QKV fusion;\ndense projections','Combined\nfrozen winner']
for i,key in enumerate(keys):
    row=summary[key]
    y=row['paired_geomean_speedup']
    low,high=row['speedup_ci95']
    final.errorbar(i,y,yerr=[[y-low],[high-y]],fmt='s',color='#0072B2',capsize=5,ms=6)
    final.text(i,y+.11,f'{y:.3f}x',ha='center',fontsize=10)
final.set(xlim=(-.5,2.5),ylim=(0,2.2),title='Held-out timing + ablations',ylabel='Speedup over paired stock eager (x)')
final.set_xticks(range(3),labels,fontsize=8)
for ax in [search,final]:
    ax.axhline(1,color='0.6',lw=.7)
    ax.axhline(1.6,color='0.3',ls='--',lw=1)
    ax.spines[['top','right']].set_visible(False)
    ax.grid(axis='y',alpha=.15)
fig.suptitle('Run 026 | Pythia-14M, BF16, B=1, T=2048, full logits | RTX 5090',fontsize=12)
fig.text(.5,.03,'Canonical R_model = 27.4827% for every row (logical opportunities, not runtime gain).\nFinal: 64 inputs x 7 paired passes x 3 processes; ablations: 1 process each. 95% crossed bootstrap intervals.',ha='center',fontsize=8)
fig.subplots_adjust(left=.07,right=.98,top=.84,bottom=.23,wspace=.33)
(RUN/'figures').mkdir(exist_ok=True)
fig.savefig(RUN/'figures/01-speedup-progress.pdf',metadata={'Title':'Run 026 speedup progress','Author':'sparsity-spillover'})
plt.close(fig)

"""Small shared visual conventions for the task.md figure set."""
import hashlib
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

HERE = Path(__file__).resolve().parent
KAPPAS = [0,.01,.05,.1,.5]
SCOPE = {'0':('#555B63','*','Dense'), '1':('#19856B','o','ReLU / 1-site'),
         '4':('#2878B5','D','4-site'), '7':('#C96024','^','7-site')}
LINES = {'none':'-', 'h':(0,(2,2)), 'all':(0,(6,3)), 'L1':'-'}
SMODEL = r'Model-wide sparsity $S_{\mathrm{model}}$ (%)'
DL = 'Loss difference from dense (nats/token)'
def setup():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.labelsize':9,
      'axes.titlesize':10,'axes.spines.top':False,'axes.spines.right':False,
      'axes.linewidth':.6,'pdf.fonttype':42,'mathtext.fontset':'dejavusans'})

def cohort(data,size):
    return [r for r in data['checkpoints'] if r['comparison_cohort'] and r['model']==size]

def pick(data,size,scope,pressure,kappa):
    matches=[r for r in cohort(data,size) if (r['scope'],r['pressure'],r['kappa'])==(scope,pressure,kappa)]
    assert len(matches)==1
    return matches[0]

def series(rows):
    for scope in ['0','1','4','7']:
        for pressure in ['none','h','all']:
            group=sorted([r for r in rows if (r['scope'],r['pressure'])==(scope,pressure)],
                         key=lambda r:-1 if r['kappa'] is None else r['kappa'])
            if group: yield scope,pressure,group

def draw_series(ax,rows,x,y,annotate=False):
    labels=[]
    for scope,pressure,group in series(rows):
        color,marker,_=SCOPE[scope]
        ax.plot([r[x] for r in group],[r[y] for r in group], color=color,marker=marker,
                ms=6 if scope=='0' else 4.5, mfc='white' if pressure=='none' else color,
                mew=.9,lw=.9,ls=LINES[pressure] if len(group)>1 else 'none',zorder=4)
        if annotate and scope in ['4','7']:
            for r in group:
                if r['kappa'] in [.05,.5]: labels.append((r[x],r[y],'.05' if r['kappa']==.05 else '.5',color))
    return labels

def label_points(ax,labels):
    """Place the small requested labels without overlapping preceding labels."""
    ax.figure.canvas.draw()
    renderer=ax.figure.canvas.get_renderer()
    boxes=[]
    offsets=[(5,7),(5,-12),(-5,8),(-5,-12),(10,17),(-10,18),(16,-18),(-16,-20),(0,24)]
    for x,y,label,color in labels:
        for dx,dy in offsets:
            text=ax.annotate(label,(x,y),xytext=(dx,dy),textcoords='offset points',
              fontsize=6.5,color=color,ha='left' if dx>=0 else 'right',va='center',
              bbox={'facecolor':'white','edgecolor':'none','pad':.25},zorder=6)
            box=text.get_window_extent(renderer).expanded(1.08,1.15)
            if ax.bbox.contains(box.x0,box.y0) and ax.bbox.contains(box.x1,box.y1) and not any(box.overlaps(b) for b in boxes):
                boxes.append(box); break
            text.remove()
        else:
            text=ax.annotate(label,(x,y),xytext=(5,5),textcoords='offset points',fontsize=6.5,color=color,zorder=6)
            boxes.append(text.get_window_extent(renderer))

def nondominated(rows,x,y,maximize_x=False):
    result=[]
    for a in rows:
        ax=-a[x] if maximize_x else a[x]
        if not any(((-b[x] if maximize_x else b[x])<=ax and b[y]<=a[y] and
                    ((-b[x] if maximize_x else b[x])<ax or b[y]<a[y])) for b in rows): result.append(a)
    return result

def outline(ax,rows,x,y):
    for r in rows:
        ax.scatter([r[x]],[r[y]],s=80,facecolors='none',edgecolors='#222222',
                   linewidths=.65,marker=SCOPE[r['scope']][1],zorder=5)

def legend(fig,clipping=False,frontier=False,y=.01):
    handles=[Line2D([],[],color=c,marker=m,mfc='white',ls='none',ms=5,label=label) for c,m,label in SCOPE.values()]
    handles += [Line2D([],[],color='#50555C',ls=LINES[p],marker='o',mfc='white' if p=='none' else '#50555C',
                       ms=4,label=label) for p,label in [('none','No pressure'),('h','OL1(h)'),('all','OL1(all)')]]
    if clipping: handles.append(Line2D([],[],color='#777777',ls=':',label='Post-hoc'))
    if frontier: handles.append(Line2D([],[],color='#222222',marker='o',mfc='none',ls='none',label='Nondominated'))
    fig.legend(handles=handles,loc='lower center',ncol=5 if len(handles)>8 else 4,
               frameon=False,fontsize=8,handlelength=2.3,columnspacing=1.5,bbox_to_anchor=(.52,y))

def polish(ax):
    ax.grid(axis='y',color='#E7E9ED',lw=.55); ax.set_axisbelow(True)
    ax.tick_params(length=3,width=.6)

def categorical(ax):
    ax.set_xticks(range(5),['0','.01','.05','.1','.5'])
    ax.tick_params(axis='x',labelbottom=True)
    ax.set_xlabel(r'Trained threshold $\kappa$'); ax.set_xlim(-.2,4.2); polish(ax)

def save(fig,name):
    path=HERE/'figures'/name
    fig.savefig(path,metadata={'Title':name.removesuffix('.pdf'),'Creator':'Analysis024 task.md figure rebuild',
                              'CreationDate':None,'ModDate':None})
    plt.close(fig)
    return {'file':path.name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}

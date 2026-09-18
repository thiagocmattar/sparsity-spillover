"""All retained settings: actual operand zeros versus matrix-instruction bypass."""
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
SOURCE = HERE/'data/operation-bypass.json'
OUTPUT = HERE/'figures/10-operation-bypass.pdf'
OPS = ['qkv_projection', 'mlp_w1', 'mlp_w2', 'attention_output_projection',
       'qk_scores', 'probability_value']
LABELS = ['QKV projection', 'FFN-up', 'FFN-down', 'Attention-output', 'QK scores', 'PV product']
COLORS = {'14M':'#3679AD', '70M':'#C27539'}
TITLE = 'Scalar Sparsity and Matrix-Instruction Bypass'


def main():
    data = json.loads(SOURCE.read_text(encoding='utf-8'))
    rows = data['settings']
    assert len(rows) == 102
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':11, 'axes.titlesize':11.5,
        'axes.labelsize':11, 'xtick.labelsize':10, 'ytick.labelsize':10,
        'axes.spines.top':False, 'axes.spines.right':False, 'axes.linewidth':.65,
        'pdf.fonttype':42, 'mathtext.fontset':'dejavusans'})
    fig, axes = plt.subplots(2, 3, figsize=(12.8, 6.7), sharex=True, sharey=True)
    fig.subplots_adjust(left=.065, right=.985, top=.87, bottom=.23, hspace=.36, wspace=.22)
    fig.suptitle(TITLE, fontsize=14, y=.975)
    fig.supylabel('Matrix instructions bypassed (%)', x=.013, fontsize=11)
    panels = []
    for i, (ax, op, label) in enumerate(zip(axes.flat, OPS, LABELS)):
        plotted = []
        for size, color in COLORS.items():
            for kind in ['clipping', 'trained']:
                group = [r for r in rows if (r['model'], r['kind']) == (size, kind)]
                xs = [100*r['operations'][op]['bf16_activation_zero_fraction_lower_bound'] for r in group]
                ys = [100*r['operations'][op]['bypass_fraction'] for r in group]
                ax.scatter(xs, ys, s=31, facecolors='white' if kind=='clipping' else color,
                           edgecolors=color if kind=='clipping' else 'white',
                           linewidths=.9 if kind=='clipping' else .6, zorder=3 if kind=='clipping' else 4)
                plotted += [{'setting_id':r['setting_id'], 'x':x, 'y':y} for r,x,y in zip(group,xs,ys)]
            for family in ['A0', 'A1-H']:
                group = sorted([r for r in rows if r['model']==size and r['kind']=='clipping' and r['family']==family], key=lambda r:r['p'])
                ax.plot([100*r['operations'][op]['bf16_activation_zero_fraction_lower_bound'] for r in group],
                        [100*r['operations'][op]['bypass_fraction'] for r in group],
                        color=color, ls=':', lw=1.1, alpha=.7, zorder=2)
            if op in ['qk_scores', 'probability_value']:
                endpoint = next(r for r in rows if r['model']==size and r['kind']=='trained'
                                and r['scope']=='7' and r['pressure']=='all' and r['kappa']==.5)
                point = endpoint['operations'][op]
                ax.annotate(r'$T_7/P_{\mathrm{all}}$, $\kappa=0.5$',
                    (100*point['bf16_activation_zero_fraction_lower_bound'], 100*point['bypass_fraction']),
                    xytext=(-7, 8 if size=='14M' else 14), ha='right', textcoords='offset points',
                    fontsize=9, color=color)
            if op == 'probability_value':
                base = next(r for r in rows if r['kind']=='trained' and r['model']==size and r['family']=='A0')
                ax.axhline(100*base['operations'][op]['bypass_fraction'], color=color, ls=(0,(4,3)), lw=.7, alpha=.55, zorder=1)
        ax.set_title(f'({chr(97+i)}) {label}', loc='left', pad=11)
        ax.set(xlim=(-3,103), ylim=(-3,103), xticks=[0,25,50,75,100], yticks=[0,25,50,75,100])
        ax.grid(axis='y', color='#E8EAED', lw=.6); ax.set_axisbelow(True)
        ax.tick_params(length=3, width=.65)
        if i >= 3:ax.set_xlabel('Scalar zero-product opportunity (%)')
        assert len(plotted) == 102
        panels.append({'operation':op, 'points':plotted})
    handles = [Line2D([],[],color=c,marker='o',ls='none',mec='white',ms=6,label=size) for size,c in COLORS.items()]
    handles += [Line2D([],[],color='#555B63',marker='o',ls='none',mec='white',ms=6,label='Trained setting'),
                Line2D([],[],color='#555B63',marker='o',ls=':',mfc='white',ms=6,label='Post-hoc clipping')]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.52,.10),ncol=4,
               frameon=False,fontsize=11,handlelength=2.3,columnspacing=2.3)
    fig.text(.52,.064,'Scalar opportunity: BF16 activation-derived lower bound. Bypass includes scalar substitution and attention padding.',
             ha='center',fontsize=9,color='#52565C')
    fig.text(.52,.035,'PV dashed guides: Base model bypass = 10.42% (14M), 5.15% (70M).',
             ha='center',fontsize=9,color='#52565C')
    fig.savefig(OUTPUT); plt.close(fig)
    evidence = {'source':SOURCE.relative_to(HERE).as_posix(), 'source_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                'script':Path(__file__).name, 'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'figure':OUTPUT.relative_to(HERE).as_posix(), 'figure_sha256':hashlib.sha256(OUTPUT.read_bytes()).hexdigest(),
                'title':TITLE, 'displayed_settings':102, 'panels':panels,
                'x':'BF16 activation-derived scalar zero-product lower bound; excludes weight zeros and P underflow',
                'y':'Raw B_op including SIMT substitution and attention padding; no baseline subtraction'}
    (HERE/'data/operation-bypass-figure.json').write_text(json.dumps(evidence,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps({'pdf':str(OUTPUT),'settings_per_panel':102,'panels':6}))


if __name__ == '__main__':
    main()

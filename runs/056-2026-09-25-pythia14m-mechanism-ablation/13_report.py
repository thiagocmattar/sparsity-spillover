"""Render audited measurements; never substitute smoke results or historical times."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from io_utils import RUN, read


def main():
    report=read(RUN/'results/mechanism-latency.json')
    if report['status']!='qualified':raise ValueError('Resolve frozen-kernel fidelity before publication')
    modes=['t00','t10','t01','t11','frozen']
    labels=['Neither','Tile only','Short-row only','Both','Frozen K050']
    rows=['# Conditional tile-bypass and short-row effects','',
          '14M T7/Pall, kappa=0.5; fixed checkpoint and gates. Three processes per mode.',
          '','| Execution | Full-model latency (ms) | Process range (ms) |','|---|---:|---:|']
    for m,label in zip(modes,labels):
        lo,hi=report['process_ranges_ms'][m]
        rows.append(f'| {label} | {report["latencies_ms"][m]:.6f} | [{lo:.6f}, {hi:.6f}] |')
    rows+=['','| Effect | Saved time (us) | Difference span (us) |','|---|---:|---:|']
    effect_names=['tile_given_short_ms','short_given_tile_ms','joint_ms','tile_alone_ms','short_alone_ms','interaction_ms']
    effect_labels=['Tile, given short rows','Short rows, given tile bypass','Joint','Tile alone','Short rows alone','Interaction']
    for name,label in zip(effect_names,effect_labels):
        lo,hi=report['difference_spans_ms'][name]
        rows.append(f'| {label} | {1000*report["effects"][name]:+.3f} | [{1000*lo:+.3f}, {1000*hi:+.3f}] |')
    rows+=['','Positive differences mean savings. Spans are process extrema, not confidence intervals.',
           'Conditional tile plus conditional short-row savings equal joint savings plus interaction.',
           'The dense fallback retains the original padded layout; it is not an optimal dense baseline.',
           'Source: `mechanism-latency.json`, audited by `07_reduce.py`.']
    (RUN/'results/mechanism-effects.md').write_text('\n'.join(rows)+'\n',encoding='utf-8')
    plt.rcParams.update({'font.family':'serif','font.size':9,'pdf.fonttype':42,
                         'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(7.1,3.05),layout='constrained')
    colors=['#969696','#4477AA','#AA3377','#228833','#555555']
    times=[report['latencies_ms'][m] for m in modes]
    axes[0].barh(labels,times,color=colors,height=.64)
    axes[0].errorbar(times,labels,xerr=[[times[i]-report['process_ranges_ms'][m][0] for i,m in enumerate(modes)],
                    [report['process_ranges_ms'][m][1]-times[i] for i,m in enumerate(modes)]],fmt='none',ecolor='black',capsize=2)
    axes[0].invert_yaxis();axes[0].set_xlabel('Full-model latency (ms)');axes[0].set_title('(a) Execution configurations',loc='left')
    names=effect_names[:3]+['interaction_ms'];labels2=['Tile | short rows','Short rows | tile','Joint','Interaction']
    values=[1000*report['effects'][n] for n in names]
    axes[1].barh(labels2,values,color=[colors[1],colors[2],colors[3],'#CCBB44'],height=.64)
    axes[1].errorbar(values,labels2,xerr=[[values[i]-1000*report['difference_spans_ms'][n][0] for i,n in enumerate(names)],
                    [1000*report['difference_spans_ms'][n][1]-values[i] for i,n in enumerate(names)]],fmt='none',ecolor='black',capsize=2)
    axes[1].axvline(0,color='black',linewidth=.65);axes[1].invert_yaxis()
    axes[1].set_xlabel('Saved time / interaction (us)');axes[1].set_title('(b) Conditional effects',loc='left')
    for ax in axes:ax.grid(axis='x',alpha=.18);ax.set_axisbelow(True)
    (RUN/'figures').mkdir(exist_ok=True)
    fig.savefig(RUN/'figures/01-mechanism-ablation.pdf');plt.close(fig)
    (RUN/'observations').mkdir(exist_ok=True)
    obs=f'''# Tile bypass and short-row execution at 14M

Question: how much do the two h/z execution mechanisms help conditionally and jointly?

Method: approved B/S factorial plus untouched K050, fixed T7/Pall kappa=0.5
checkpoint and gates, one RTX5090, BF16 B1/T2048 full-logit inference. Fifteen
fresh processes; 64 fixed timing inputs, seven paired passes. All 338 validation
blocks from 500 documents; 1444-token tail excluded. Full logits and h/z operands
match frozen K050 bitwise; all numerical gates and work-count checks pass.

Caption: (a) Full-model latency of four execution combinations and frozen K050.
(b) Conditional tile and short-row effects, joint savings and interaction.
Bars use geometric-mean host latency. Whiskers show process extrema and derived
difference spans, not confidence intervals. Positive values denote savings;
the interaction is not a third independently measured source of saved time.

Result: tile given short rows = {report['effects']['tile_given_short_ms']*1000:+.3f} us;
short rows given tile = {report['effects']['short_given_tile_ms']*1000:+.3f} us;
joint = {report['effects']['joint_ms']*1000:+.3f} us;
interaction = {report['effects']['interaction_ms']*1000:+.3f} us.
See [the complete table](../results/mechanism-effects.md) and
[audited measurements](../results/mechanism-latency.json) for signs and variability.

Caveats: one checkpoint/device/workload. Whole-group completion belongs to the
short-row mechanism, including bias-only zero rows. Shared inspection, scheduling,
and padded fallback costs remain in the net effects. These effects do not uniquely
partition total savings, memory time, arithmetic time, or the PyTorch gap.

Source script: `13_report.py`; measurement audit: `07_reduce.py`.
'''
    (RUN/'observations/001-mechanism-ablation.md').write_text(obs,encoding='utf-8')
    (RUN/'observations/INDEX.md').write_text('# Observations\n\n- [001: conditional tile/short-row effects](001-mechanism-ablation.md).\n',encoding='utf-8')


if __name__=='__main__':main()

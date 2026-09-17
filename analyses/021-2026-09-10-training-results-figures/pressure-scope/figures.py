"""Pressure-scope figure alternatives from the audited, retained measurements."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
import numpy as np

from evidence import HERE, ANALYSIS, KAPPAS, FAMILIES, LABELS, SITES, module

BLUE, ORANGE = "#2878B5", "#C96024"
OPS = ("qkv_projection", "qk_scores", "probability_value", "attention_output_projection", "mlp_w1", "mlp_w2")
OP_LABELS = ("QKV", "QK", "PV", "Attention output", "FFN up", "FFN down")
OP_COLORS = ("#9eaec2", "#df985c", "#f0c68b", "#76969c", "#817ca9", "#b5acd0")
STYLE = {"none": ("o", "-", "white"), "h": ("D", "--", "white"), "all": ("^", "-.", None)}


def setup():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.labelsize": 8,
                         "axes.titlesize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
                         "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42,
                         "axes.linewidth": .6})


def scope(family):
    return "h" if family.endswith("-H") else "all" if "OL1" in family else "none"


def color(family):
    return BLUE if family.startswith("A4") else ORANGE


def rows(data, family, kappa=None):
    return sorted([r for r in data["rows"] if r["family"] == family and (kappa is None or r["kappa"] == kappa)], key=lambda r: r["kappa"])


def line(ax, x, y, family, label=None):
    marker, ls, face = STYLE[scope(family)]
    ax.plot(x, y, marker=marker, ls=ls, color=color(family), lw=.9, ms=4.5,
            mfc=face or color(family), mew=.8, label=label or LABELS[family], zorder=3)


def grid(ax):
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#ececec", lw=.5)


def overview(data):
    setup()
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    fig.subplots_adjust(left=.095, right=.98, bottom=.27, top=.95)
    controls = module("01_quality_sparsity").read_evidence()
    for fam, c in (("A0", "#777777"), ("A1-H", "#88a799")):
        points = sorted([r for r in controls["points"] if r["family"] == fam and r["kind"] == "clipped"], key=lambda r: r["dose"])
        ax.plot([100*r["R_model"] for r in points], [r["loss"] for r in points], ":o", color=c, alpha=.6, ms=2.5, mfc="white", lw=.7)
        p = next(r for r in data["original_trained"] if r["scale"] == "14M" and r["family"] == fam)
        ax.scatter(100*p["R_model"], p["loss"], color=c, s=20, zorder=4)
        ax.annotate("Baseline" if fam == "A0" else "ReLU", (100*p["R_model"], p["loss"]),
                    xytext=(5, -9 if fam == "A0" else 5), textcoords="offset points", fontsize=7, color=c)
    for family in FAMILIES:
        selected = rows(data, family)
        line(ax, [r["S_model_percent"] for r in selected], [r["loss"] for r in selected], family)
    for fam, x in (("A4", 12.833), ("A7", 29.952)):
        ax.axvline(x, color=color(fam), lw=.8, alpha=.4)
        ax.text(x-.35, 6.18, f'{fam[1]}-site ceiling: {x:.2f}%', ha="right", fontsize=7, color=color(fam))
    ax.text(.7, 5.99, "Post-hoc thresholding", color="#888888", fontsize=7)
    ax.annotate("OL1(h), $\\kappa=0.05$", (10.1261, 5.19496), xytext=(12.6, 5.12), fontsize=7, color=ORANGE,
                arrowprops={"arrowstyle": "-", "lw": .5, "color": ORANGE})
    ax.annotate("27.48%", (27.4827, 5.8294), xytext=(-1, 8), textcoords="offset points", ha="center", fontsize=7, color=ORANGE)
    ax.set(xlim=(-.4, 31), ylim=(5.07, 6.23), xlabel=r"Model-wide sparsity $\mathcal{S}_{\mathrm{model}}$ (%)", ylabel="Validation loss (lower is better)")
    grid(ax)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.54, .025), ncol=2, frameon=False, fontsize=7, handlelength=2.5, columnspacing=2)
    return fig


def paired(data):
    setup()
    fig, axes = plt.subplots(2, 2, sharex=True, sharey="row", figsize=(7.2, 3.65))
    fig.subplots_adjust(left=.1, right=.98, top=.89, bottom=.25, hspace=.17, wspace=.2)
    for col, prefix in enumerate(("A4", "A7")):
        for treatment, reference, label in ((prefix+"-OL1-H", prefix, r"$P_h-P_0$"), (prefix+"-OL1", prefix+"-OL1-H", r"$P_{\mathrm{all}}-P_h$")):
            selected = [r for r in data["contrasts"] if (r["treatment"], r["reference"]) == (treatment, reference)]
            for ax, key in zip(axes[:, col], ("delta_loss", "delta_s_pp")):
                line(ax, np.arange(5), [r[key] for r in selected], treatment, label)
                ax.axhline(0, color="#888888", lw=.7, ls=":")
                grid(ax)
        axes[0,col].set_title(f'({"ab"[col]}) {prefix[1]}-site thresholds', loc="left", pad=7)
        axes[1,col].set(xlabel=r"Threshold $\kappa$", xticks=np.arange(5), xticklabels=[f"{k:g}" for k in KAPPAS])
    axes[0,0].set(ylim=(-.31,.36), ylabel=r"$\Delta$ validation loss")
    axes[1,0].set(ylim=(-.8,12.2), ylabel=r"$\Delta\mathcal{S}_{\mathrm{model}}$ (pp)")
    fig.text(.54,.975,r"Matched pressure-scope changes: $P_0\rightarrow P_h\rightarrow P_{\mathrm{all}}$", ha="center", va="top", fontsize=8)
    handles = [Line2D([],[], color="#555555", marker=STYLE[s][0], ls=STYLE[s][1], mfc=STYLE[s][2] or "#555555", lw=.9, ms=4,
                      label=label) for s,label in (("h", r"$P_h-P_0$: add h-only pressure"), ("all", r"$P_{\mathrm{all}}-P_h$: broaden pressure"))]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.54,.015), ncol=2, frameon=False, fontsize=7)
    return fig


def geometry(data):
    setup()
    fig, axes = plt.subplots(1,3, figsize=(7.6, 2.8), gridspec_kw={"width_ratios": [1,1,.85]})
    fig.subplots_adjust(left=.085, right=.98, top=.87, bottom=.29, wspace=.42)
    families = [f for f in FAMILIES if "OL1" in f]
    for i,f in enumerate(families):
        selected = sorted([r for r in data["geometry"] if r["family"] == f], key=lambda r:r["kappa"])
        for ax,key,mult in ((axes[0],"rho",100),(axes[1],"ratio",1)):
            med = np.array([np.median([s[key] for s in r["steps"]]) for r in selected]) * mult
            line(ax, np.arange(5)+(i-1.5)*.045, med, f)
        g = data["geometry_groups"][f]
        axes[2].plot([0,g["cap_percent"]],[3-i,3-i], color=color(f), lw=.8)
        axes[2].plot(g["cap_percent"],3-i, STYLE[scope(f)][0], mfc=STYLE[scope(f)][2] or color(f), mec=color(f), ms=4.5)
        axes[2].text(max(g["cap_percent"]+2, 3),3-i,f'{g["cap_percent"]:.2f}%', va="center", fontsize=6.8, color=color(f))
    for ax,title in zip(axes,("(a) Opposing component", "(b) Pre-cap norm ratio", "(c) Cap activity")):
        ax.set_title(title,loc="left",pad=9)
    for ax in axes[:2]:
        ax.set(yscale="log", xticks=np.arange(5), xticklabels=[f"{k:g}" for k in KAPPAS], xlabel=r"Threshold $\kappa$")
        grid(ax)
    axes[0].set_ylabel(r"Removed component (% of $\|u\|$)")
    axes[1].set_ylabel(r"Median $r/b$")
    axes[1].axhline(1,color="#777777",ls=":",lw=.7)
    axes[2].set(xlim=(0,125), xticks=[0,50,100], yticks=range(4), yticklabels=["7 / all", "7 / h", "4 / all", "4 / h"], xlabel="Steps capped (%)")
    handles=[Line2D([],[], color=color(f), marker=STYLE[scope(f)][0], ls=STYLE[scope(f)][1], mfc=STYLE[scope(f)][2] or color(f),lw=.9,ms=4,label=LABELS[f]) for f in families]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.54,.015), ncol=4,frameon=False,fontsize=6.5,handlelength=2.2,columnspacing=1)
    return fig


def structure(data,kappa):
    setup()
    fig, axes = plt.subplots(1,3,figsize=(7.4,3.1),gridspec_kw={"width_ratios":[.95,.95,1.6]})
    fig.subplots_adjust(left=.055,right=.985,top=.84,bottom=.27,wspace=.25)
    for ax,fs,letter in ((axes[0],FAMILIES[:3],"a"),(axes[1],FAMILIES[3:],"b")):
        selected = [rows(data,f,kappa)[0] for f in fs]
        matrix=np.array([[r["sites"][s]["percent"] for r in selected] for s in SITES])
        ax.imshow(matrix,vmin=0,vmax=100,cmap="Greys",aspect="auto")
        for (i,j),v in np.ndenumerate(matrix):
            ax.text(j,i,f"{v:.1f}",ha="center",va="center",fontsize=6.6,color="white" if v>55 else "black")
        ax.set(xticks=range(3),xticklabels=[r"$P_0$",r"$P_h$",r"$P_{\mathrm{all}}$"],yticks=range(7),yticklabels=[r"$a$",r"$m$",r"$h$",r"$q$",r"$k$",r"$v$",r"$z$"],xlabel="Pressure scope")
        ax.set_title(f"({letter}) {fs[0][1]}-site: zeros (%)",loc="left",color=color(fs[0]),pad=8)
        ax.tick_params(length=0)
    ax=axes[2]
    selected=[rows(data,f,kappa)[0] for f in FAMILIES]
    x=np.array([0,1,2,3.6,4.6,5.6]);bottom=np.zeros(6)
    for op,label,c in zip(OPS,OP_LABELS,OP_COLORS):
        vals=np.array([r["operations_pp"][op] for r in selected])
        ax.bar(x,vals,bottom=bottom,color=c,width=.7,label=label,edgecolor="white",linewidth=.2)
        bottom+=vals
    for xi,v in zip(x,bottom):ax.text(xi,v+.4,f"{v:.2f}%",ha="center",fontsize=6.3)
    ax.set(ylim=(0,32 if kappa==.5 else 13),xticks=x,xticklabels=[r"$P_0$",r"$P_h$",r"$P_{\mathrm{all}}$"]*2,ylabel=r"$\mathcal{S}_{\mathrm{model}}$ (pp)")
    ax.set_title("(c) Operation contributions",loc="left",pad=8)
    ax.text(1,-.19,"4-site",color=BLUE,ha="center",transform=ax.get_xaxis_transform(),fontsize=7)
    ax.text(4.6,-.19,"7-site",color=ORANGE,ha="center",transform=ax.get_xaxis_transform(),fontsize=7)
    grid(ax)
    fig.legend(*ax.get_legend_handles_labels(),loc="lower center",bbox_to_anchor=(.52,.012),ncol=6,frameon=False,fontsize=6.5,handlelength=1,columnspacing=1)
    fig.text(.52,.98,rf"Pythia-14M, $\kappa={kappa:g}$",ha="center",va="top",fontsize=8)
    return fig


def layer_maps(data,kappa):
    setup()
    fig,axes=plt.subplots(2,3,figsize=(7.3,4.5),sharex=True,sharey=True)
    fig.subplots_adjust(left=.065,right=.98,top=.89,bottom=.12,hspace=.3,wspace=.18)
    for ax,f in zip(axes.flat,FAMILIES):
        r=rows(data,f,kappa)[0]
        a=np.array([[100*p["zero"]/p["total"] for p in sorted(r["layers"][s],key=lambda p:p["layer"])] for s in SITES])
        ax.imshow(a,vmin=0,vmax=100,cmap="Greys",aspect="auto")
        for (i,j),v in np.ndenumerate(a):ax.text(j,i,f"{v:.0f}",ha="center",va="center",fontsize=6,color="white" if v>55 else "black")
        ax.set(xticks=range(6),xticklabels=range(1,7),yticks=range(7),yticklabels=["a","m","h","q","k","v","z"])
        ax.set_title(LABELS[f],color=color(f),fontsize=7)
        ax.tick_params(length=0)
    for ax in axes[1]:ax.set_xlabel("Layer")
    fig.suptitle(rf"Site-by-layer exact zeros (%), $\kappa={kappa:g}$",fontsize=9)
    return fig


def scale(data):
    mod=module("05_scale_transfer")
    mod.STYLE={"A4-OL1":("4-site + OL1(all)",BLUE,"^"),"A7-OL1":("7-site + OL1(all)",ORANGE,"^")}
    fig=mod.make_figure(mod.read_evidence())
    for ax in fig.axes:
        for l in ax.lines:
            if str(l.get_gid()).endswith(("A4-OL1","A7-OL1")):l.set_linestyle("-.")
    for t in fig.findobj(matplotlib.text.Text):
        t.set_text(t.get_text().replace("post-hoc clipping","post-hoc thresholding"))
    # Shared STYLE supplies the marker encoding for future h-only endpoints.
    # No pending 70M measurements or inferred points are inserted.
    return fig


def kernel(data):
    setup()
    fig,(a,b)=plt.subplots(1,2,figsize=(7.4,3.1),gridspec_kw={"width_ratios":[1.25,1]})
    fig.subplots_adjust(left=.085,right=.98,top=.86,bottom=.29,wspace=.28)
    runtime=data["runtime"]
    for family,c,marker,label in (("baseline/local","#888888","o","Baseline / 1-site"),("4-site",BLUE,"D","4-site"),("7-site",ORANGE,"^","7-site")):
        selected=[r for r in runtime if r["family"]==family]
        faces=[c if family=="baseline/local" or r["pressure"]=="orthogonal_l1" else "white" for r in selected]
        a.scatter([r["full_candidate_gm_ms"] for r in selected],[r["validation_loss"] for r in selected],c=faces,edgecolors=c,marker=marker,s=25,linewidths=.8,label=label,zorder=3)
        b.scatter([100*r["projection_mma_bypass_fraction"] for r in selected],[r["projection_sparse_gain"] for r in selected],c=faces,edgecolors=c,marker=marker,s=25,linewidths=.8,zorder=3)
    extra=data["h_only_runtime"]
    a.scatter([r["full_candidate_gm_ms"] for r in extra], [r["validation_loss"] for r in extra],
              marker="s",s=28,facecolors="white",edgecolors=BLUE,linewidths=1,label="4-site + OL1(h)",zorder=4)
    b.scatter([100*r["projection_mma_bypass_fraction"] for r in extra], [r["projection_sparse_gain"] for r in extra],
              marker="s",s=28,facecolors="white",edgecolors=BLUE,linewidths=1,zorder=4)
    runtime = runtime + extra
    x=np.array([100*r["projection_mma_bypass_fraction"] for r in runtime]);y=np.array([r["projection_sparse_gain"] for r in runtime])
    slope,intercept=np.polyfit(x,y,1);line_x=np.linspace(0,85,80)
    b.plot(line_x,intercept+slope*line_x,"--",color="#666666",lw=1)
    b.text(.08,.87,rf'$R^2={np.corrcoef(x,y)[0,1]**2:.3f}$',transform=b.transAxes,fontsize=8)
    base=next(r for r in runtime if r["condition"]=="c01")
    a.scatter(base["native_baseline_gm_ms"],base["validation_loss"],marker="x",color="black",s=28,zorder=5)
    a.annotate("Native A0",(base["native_baseline_gm_ms"],base["validation_loss"]),xytext=(-5,-12),textcoords="offset points",ha="right",fontsize=7)
    for condition,text,offset in (("c20","4-site + OL1(all)",(-4,8)),("c30","7-site + OL1(all)",(5,-15))):
        r=next(r for r in runtime if r["condition"]==condition)
        a.annotate(text,(r["full_candidate_gm_ms"],r["validation_loss"]),xytext=offset,textcoords="offset points",fontsize=6.8,color=BLUE if condition=="c20" else ORANGE)
    a.set(title="(a) Quality and absolute latency (35 models)",xlabel="Full-model K050 latency (ms)",ylabel="Validation loss",xlim=(.435,.68),ylim=(5.04,6.15))
    b.set(title="(b) Projection MMA bypass (35 models)",xlabel="Projection MMA bypass (%)",ylabel="Projection-skipping gain (×)",xlim=(-2,86),ylim=(.95,1.405))
    b.axhline(1,color="#999999",lw=.7,ls=":")
    for ax in (a,b):grid(ax)
    handles,labels=a.get_legend_handles_labels()
    fig.legend(handles,labels,loc="lower center",bbox_to_anchor=(.52,.07),ncol=4,frameon=False,fontsize=6.8)
    fig.text(.52,.015,"Multisite diamonds/triangles: open = no pressure; filled = OL1(all). Squares = OL1(h).",ha="center",fontsize=6.5)
    return fig


if __name__=="__main__":
    data=json.loads((HERE/"data/evidence.json").read_text())
    out=HERE/"figures";out.mkdir(exist_ok=True)
    builders={"01-overview.pdf":overview,"03-pressure-increments.pdf":paired,"04-ol1-geometry.pdf":geometry,
              "05-site-structure.pdf":lambda d:structure(d,.5),"06-scale.pdf":scale,"07-quality-latency.pdf":kernel,
              "appendix-moderate-structure.pdf":lambda d:structure(d,.05)}
    for name,builder in builders.items():
        fig=builder(data);fig.savefig(out/name,metadata={"CreationDate":None,"Creator":"Analysis 021 pressure-scope/figures.py"});plt.close(fig);print(name)
    with PdfPages(out/"appendix-layer-zeros.pdf",metadata={"CreationDate":None}) as pdf:
        for k in (.05,.5):
            fig=layer_maps(data,k);pdf.savefig(fig);plt.close(fig)
    print("appendix-layer-zeros.pdf (two pages)")

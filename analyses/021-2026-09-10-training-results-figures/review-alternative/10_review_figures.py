"""Retained-evidence figures for the September manuscript review corrections."""
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.text import Text
import numpy as np

HERE = Path(__file__).resolve().parent
ANALYSIS = HERE.parent
ROOT = ANALYSIS.parents[1]
BLUE, ORANGE, GRAY = "#2878B5", "#C96024", "#777777"


def module(name):
    spec = importlib.util.spec_from_file_location(name, ANALYSIS / f"{name}.py")
    out = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(out)
    return out


def read_evidence():
    endpoint_path = ROOT / "analyses/018-2026-09-08-results-materials/figure_data.json"
    runtime_path = ANALYSIS / "investigation/data/checkpoints.json"
    endpoints = json.loads(endpoint_path.read_text(encoding="utf-8"))
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    assert len(runtime) == 30 and {r["condition"] for r in runtime} == {f"c{i:02d}" for i in range(1, 31)}
    sources = [endpoint_path, runtime_path]
    density = []
    for family, dose in (("A0", None), ("A4-OL1", .5), ("A7-OL1", .5)):
        path = ROOT / f"runs/031-2026-09-08-signed-activation-density/results/histograms/14M_{family}_{dose}.json.gz"
        sources.append(path)
        r = json.loads(gzip.decompress(path.read_bytes()))
        source = next(p for p in endpoints["trained"] if p["id"] == r["source"]["id"])
        assert source["family"] == family and source["identity"] == r["source"]["identity"]
        assert r["coverage"]["sequences"] == 338 and r["coverage"]["excluded_tail_tokens"] == 1444
        groups = {}
        for label, group in (("m", r["sites"]["m"]), ("h", r["sites"]["h"]), ("qkv", r["groups"]["Attention activations"])):
            counts = np.asarray(group["histogram"], dtype=np.int64)
            assert counts.sum() + group["exact_zero_count"] + group["underflow"] + group["overflow"] == group["total"]
            assert group["nonfinite"] == 0
            edges = np.linspace(r["grid"]["lower"], r["grid"]["upper"], r["grid"]["bins"] + 1, dtype=np.float32)[::10].astype(float)
            groups[label] = {k: group[k] for k in ("total", "exact_zero_count", "underflow", "overflow")}
            groups[label].update(bin_counts=counts.reshape(-1, 10).sum(axis=1).tolist(), zero_percent=100*group["exact_zero_count"]/group["total"])
        layers = []
        for row in r["rows"]:
            layers.append({k: row[k] for k in ("name", "total", "exact_zero_count")})
        for site in ("m", "h", "q_post", "k_post", "v"):
            selected = [p for p in layers if p["name"].startswith(site + ".")]
            assert len(selected) == 6
            assert sum(p["exact_zero_count"] for p in selected) == r["sites"][site]["exact_zero_count"]
        density.append({"family": family, "source_id": source["id"], "groups": groups, "layers": layers})
    operations = []
    for family in ("A4", "A4-OL1", "A7", "A7-OL1"):
        r, = [p for p in endpoints["trained"] if p["scale"] == "14M" and p["family"] == family and p["dose"] == .5]
        counts = r["counts"]
        contribution = {k: 100*v["zero_product_count"]/counts["model_product_count"] for k,v in counts["per_operation"].items()}
        assert np.isclose(sum(contribution.values()), 100*r["R_model"])
        operations.append({"family": family, "source_id": r["id"], "loss": r["loss"], "s_model_percent": 100*r["R_model"], "contributions_pp": contribution})
    p4, p7 = operations[1], operations[3]
    interaction = {"sparsity_pp": (p7["s_model_percent"]-operations[2]["s_model_percent"])-(p4["s_model_percent"]-operations[0]["s_model_percent"]),
                   "loss": (p7["loss"]-operations[2]["loss"])-(p4["loss"]-operations[0]["loss"])}
    scales = []
    for scale in ("14M", "70M", "410M"):
        a0, = [p for p in endpoints["trained"] if p["scale"] == scale and p["family"] == "A0"]
        a7, = [p for p in endpoints["trained"] if p["scale"] == scale and p["family"] == "A7-OL1" and p["dose"] == .5]
        delta = a7["loss"]-a0["loss"]
        scales.append({"scale": scale, "dense_loss": a0["loss"], "seven_site_loss": a7["loss"], "delta_loss": delta, "perplexity_ratio": float(np.exp(delta))})
    return {"sources": {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
            "density": density, "edges": edges.tolist(), "operations": operations, "runtime": runtime,
            "interaction_at_kappa_0_5": interaction, "scale_quality_costs": scales}


def style():
    plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":8, "axes.labelsize":8,
        "axes.titlesize":8, "xtick.labelsize":7, "ytick.labelsize":7, "pdf.fonttype":42,
        "axes.spines.top":False,"axes.spines.right":False,"axes.linewidth":.6,"axes.axisbelow":True})


def densities(data):
    style()
    base = module("04_distributions_and_operations")
    fig = plt.figure(figsize=(7.4, 3.5))
    gs = fig.add_gridspec(1,4,width_ratios=[1,1,1,2.15],left=.075,right=.985,bottom=.31,top=.81,wspace=.48)
    axes = [fig.add_subplot(gs[0,i]) for i in range(4)]
    fig.text(.075,.97,r"(a) Site-specific activations ($\kappa=0.5$, 14M)",va="top",fontsize=9)
    fig.text(axes[3].get_position().x0,.97,"(b) Operation contributions",va="top",fontsize=9)
    edges = np.array(data["edges"])
    for ax, key, title in zip(axes[:3], ("m","h","qkv"), (r"FFN up input $m$",r"FFN hidden $h$",r"Attention $q,k,v$")):
        for record, color in zip(data["density"], (GRAY,BLUE,ORANGE)):
            g = record["groups"][key]
            ax.stairs(np.array(g["bin_counts"])/(g["total"]*np.diff(edges)),edges,color=color,lw=.8 if color==GRAY else 1.,alpha=.65 if color==GRAY else 1)
        ax.set_title(title,pad=8)
        ax.set_yscale("symlog",linthresh=.01,linscale=.5)
        ax.set(ylim=(0,50),xlim=(-4,4) if key=="qkv" else (-2.5,3),xlabel="Activation")
        ax.set_xticks([-4,0,4] if key=="qkv" else [-2,0,2])
        ax.set_yticks([0,.01,.1,1,10],["0",".01",".1","1","10"])
        ax.grid(color=".93",lw=.4)
        for i, (r,c) in enumerate(zip(data["density"], (GRAY,BLUE,ORANGE))):
            name = ("Dense","4-site + OL1","7-site + OL1")[i]
            ax.text(0,-.36-i*.09,f'{name}: {r["groups"][key]["zero_percent"]:.2f}% zeros',transform=ax.transAxes,fontsize=7,color=c)
    axes[0].set_ylabel("Nonzero density")
    for ax in axes[1:3]: ax.tick_params(labelleft=False)
    ax=axes[3]; bottom=np.zeros(4)
    for key,label,color in base.OPERATIONS:
        vals=np.array([p["contributions_pp"][key] for p in data["operations"]])
        ax.bar(range(4),vals,bottom=bottom,width=.64,color=color,edgecolor="white",lw=.3,label=label)
        bottom+=vals
    for i,p in enumerate(data["operations"]):
        ax.text(i,bottom[i]+.6,f'{bottom[i]:.1f}%',ha="center",fontsize=7)
    ax.set(ylim=(0,31),ylabel=r"Contribution to $\mathcal{S}_{\mathrm{model}}$ (pp)")
    ax.set_xticks(range(4),["4-site\nno pressure","4-site\n+ OL1","7-site\nno pressure","7-site\n+ OL1"],fontsize=6.5)
    ax.grid(axis="y",color=".93",lw=.4)
    ax.legend(loc="upper center",bbox_to_anchor=(.5,-.30),ncol=3,frameon=False,fontsize=6.5,handlelength=1,handletextpad=.3,columnspacing=.65)
    return fig


def heatmap(data):
    style()
    fig,axes=plt.subplots(1,3,figsize=(7.4,2.7),sharey=True)
    fig.subplots_adjust(left=.12,right=.89,bottom=.20,top=.85,wspace=.18)
    sites=("m","h","q_post","k_post","v")
    for ax,record,title in zip(axes,data["density"],("Dense reference","4-site + OL1","7-site + OL1")):
        lookup={p["name"]:p for p in record["layers"]}
        values=np.array([[100*lookup[f'{s}.layer_{l}']["exact_zero_count"]/lookup[f'{s}.layer_{l}']["total"] for l in range(6)] for s in sites])
        im=ax.imshow(values,vmin=0,vmax=100,cmap="Blues",aspect="auto")
        ax.set_title(title);ax.set_xticks(range(6));ax.set_xlabel("Layer")
        ax.set_yticks(range(5),["m: FFN up","h: FFN hidden","q","k","v"])
        for i in range(5):
            for j in range(6): ax.text(j,i,f'{values[i,j]:.0f}',ha="center",va="center",fontsize=6,color="white" if values[i,j]>55 else "black")
    cax=fig.add_axes((.91,.20,.018,.65));fig.colorbar(im,cax=cax,label="Exact zeros (%)")
    return fig


def kernel(data):
    style()
    fig,(a,b)=plt.subplots(1,2,figsize=(7.4,3.15))
    fig.subplots_adjust(left=.075,right=.985,bottom=.24,top=.84,wspace=.28)
    styles={"baseline/local":(GRAY,"o","Dense / one-site"),"4-site":(BLUE,"D","4-site"),"7-site":(ORANGE,"^","7-site")}
    for family,(color,marker,label) in styles.items():
        rows=[p for p in data["runtime"] if p["family"]==family]
        faces=[color if (p["pressure"]=="orthogonal_l1" or family=="baseline/local") else "white" for p in rows]
        for ax,xkey,ykey,scale in ((a,"projection_on_candidate_gm_ms","validation_loss",1),(b,"projection_mma_bypass_fraction","projection_sparse_gain",100)):
            ax.scatter([scale*p[xkey] for p in rows],[p[ykey] for p in rows],s=26,marker=marker,c=faces,edgecolors=color,lw=.7,zorder=3)
    a.set(title="(a) Absolute latency and validation loss",xlabel="Projection-only K050 latency (ms)",ylabel="Validation loss",xlim=(.44,.66),ylim=(5.0,6.15))
    a.text(.02,.04,"Lower / left is better",transform=a.transAxes,color=".45",fontsize=6.5)
    for condition,offset in (("c20",(5,-12)),("c30",(8,5))):
        p=next(p for p in data["runtime"] if p["condition"]==condition)
        a.annotate(f'{p["family"]} + OL1',xy=(p["projection_on_candidate_gm_ms"],p["validation_loss"]),xytext=offset,textcoords="offset points",fontsize=7,color=BLUE if condition=="c20" else ORANGE)
    dense=next(p for p in data["runtime"] if p["condition"]=="c01")
    a.scatter([dense["native_baseline_gm_ms"]],[dense["validation_loss"]],marker="x",s=27,color="black",zorder=4)
    a.annotate("Native dense",(dense["native_baseline_gm_ms"],dense["validation_loss"]),xytext=(-3,13),textcoords="offset points",ha="right",fontsize=6.5)
    x=np.array([100*p["projection_mma_bypass_fraction"] for p in data["runtime"]]);y=np.array([p["projection_sparse_gain"] for p in data["runtime"]])
    coef=np.polyfit(x,y,1);xx=np.linspace(x.min(),x.max(),100)
    b.plot(xx,np.polyval(coef,xx),ls="--",color=".3",lw=1)
    b.text(.06,.87,rf'$R^2={np.corrcoef(x,y)[0,1]**2:.3f}$',transform=b.transAxes)
    b.axhline(1,color=".6",ls=":",lw=.8)
    b.set(title="(b) Zero fragments and projection gain",xlabel="Projection MMA bypass (%)",ylabel="Projection-skipping gain (×)",xlim=(-2,85),ylim=(.95,1.41))
    for ax in (a,b):ax.grid(axis="y",color=".93",lw=.4)
    handles=[Line2D([],[],color=c,marker=m,ls="none",label=l,ms=4) for c,m,l in styles.values()]
    fig.legend(handles=handles,ncol=3,loc="lower center",bbox_to_anchor=(.5,.01),frameon=False,fontsize=7)
    return fig


def terminology_figure(name):
    m=module(name);evidence=m.read_evidence();fig=m.make_figure(evidence)
    if isinstance(fig,tuple):fig=fig[0]
    for text in fig.findobj(Text):
        value=text.get_text()
        value=value.replace("Post-hoc clipping","Post-hoc thresholds").replace("A0 + post-hoc clipping","Dense + post-hoc thresholds")
        value=value.replace("4-site ceiling","4-site reach").replace("7-site ceiling","7-site reach").replace("of ceiling","of reach")
        if value=="Baseline":value="Dense"
        text.set_text(value)
    return fig,evidence


def main():
    data=read_evidence()
    out=HERE/"figures";out.mkdir(exist_ok=True)
    figures={"04-site-distributions.pdf":densities(data),"appendix-site-zero-heatmap.pdf":heatmap(data),"07-kernel-quality-latency.pdf":kernel(data)}
    for name,builder in (("01-quality-sparsity.pdf","01_quality_sparsity"),("05-cross-size.pdf","05_scale_transfer")):
        figures[name],sidecar=terminology_figure(builder)
        data[builder+"_evidence"]=sidecar
    for name,fig in figures.items():
        fig.savefig(out/name,metadata={"Creator":"Analysis 021 / 10_review_figures.py","CreationDate":None})
        plt.close(fig)
    (HERE/"data/review-corrections.json").write_text(json.dumps(data,indent=2,ensure_ascii=False)+"\n",encoding="utf-8",newline="\n")
    print(json.dumps({"site_zeros":{r["family"]:{k:v["zero_percent"] for k,v in r["groups"].items()} for r in data["density"]},"interaction":data["interaction_at_kappa_0_5"],"scale_costs":data["scale_quality_costs"]},indent=2))


if __name__=="__main__":main()

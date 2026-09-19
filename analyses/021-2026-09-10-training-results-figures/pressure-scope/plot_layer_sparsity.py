"""Restyle the retained 14M layer maps for author review; no manuscript inclusion."""
import hashlib
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SOURCE = HERE / "data/evidence.json"
OUTPUT = HERE / "figures/appendix-14m-layer-sparsity.pdf"
REVIEW = ROOT / "manuscript/draft/figures" / OUTPUT.name
SITES = ("a", "m", "h", "q_post", "k_post", "v", "z")
RECIPES = (
    ("A4", r"$T_4/P_0$"),
    ("A4-OL1-H", r"$T_4/P_h$"),
    ("A4-OL1", r"$T_4/P_{\mathrm{all}}$"),
    ("A7", r"$T_7/P_0$"),
    ("A7-OL1-H", r"$T_7/P_h$"),
    ("A7-OL1", r"$T_7/P_{\mathrm{all}}$"),
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    data = json.loads(SOURCE.read_text(encoding="utf-8"))
    old_pdf = ROOT / "manuscript/draft/figures/pressure-scope/appendix-layer-zeros.pdf"
    preserved = {p: sha(p) for p in (old_pdf, ROOT / "manuscript/draft/results-appendix.tex")}
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.labelsize": 11, "axes.titlesize": 12,
                         "xtick.labelsize": 10, "ytick.labelsize": 10,
                         "pdf.fonttype": 42, "mathtext.fontset": "dejavusans"})
    cmap = plt.get_cmap("viridis")
    pages = []
    with PdfPages(OUTPUT, metadata={"Title": "Pythia-14M: activation sparsity by site and layer",
                                   "CreationDate": None, "ModDate": None}) as pdf:
        for kappa in (.05, .5):
            fig, axes = plt.subplots(2, 3, figsize=(10.6, 6.2), sharex=True, sharey=True)
            fig.subplots_adjust(left=.073, right=.89, bottom=.12, top=.83,
                                wspace=.18, hspace=.38)
            fig.suptitle("Pythia-14M: activation sparsity by site and layer", fontsize=15, y=.975)
            fig.text(.48, .915, rf"Threshold $\kappa={kappa:g}$", ha="center", fontsize=12)
            panels = []
            for letter, ax, (family, label) in zip("abcdef", axes.flat, RECIPES):
                selected = [r for r in data["rows"] if r["family"] == family and r["kappa"] == kappa]
                assert len(selected) == 1
                row = selected[0]
                source = Path(row["source_attempt"]) / "diagnostics/activation_statistics.json"
                assert sha(ROOT / source) == data["sources"][source.as_posix()]
                raw = json.loads((ROOT / source).read_text(encoding="utf-8"))
                cells = []
                for site in SITES:
                    counts = sorted(row["layers"][site], key=lambda p: p["layer"])
                    assert [p["layer"] for p in counts] == list(range(6))
                    for p in counts:
                        original = next(r for r in raw["rows"] if r["name"] == f"{site}.layer_{p['layer']}")
                        assert (p["zero"], p["total"]) == (original["exact_zero_count"], original["total"])
                        assert 0 <= p["zero"] <= p["total"] and p["total"] > 0
                    cells.append([100 * p["zero"] / p["total"] for p in counts])
                values = np.asarray(cells)
                im = ax.imshow(values, vmin=0, vmax=100, cmap=cmap, aspect="auto", interpolation="nearest")
                for (i, j), value in np.ndenumerate(values):
                    ax.text(j, i, f"{value:.0f}", ha="center", va="center", fontsize=9,
                            color="black", path_effects=[
                                path_effects.withStroke(linewidth=1.1, foreground=(1, 1, 1, .8))])
                ax.set(xticks=range(6), xticklabels=range(1, 7),
                       yticks=range(7), yticklabels=[rf"${s}$" for s in ("a", "m", "h", "q", "k", "v", "z")])
                ax.set_title(f"({letter}) {label}", color="black", pad=9)
                ax.set_xticks(np.arange(-.5, 6, 1), minor=True)
                ax.set_yticks(np.arange(-.5, 7, 1), minor=True)
                ax.grid(which="minor", color="white", alpha=.28, linewidth=.5)
                ax.tick_params(which="both", length=0, pad=5)
                for spine in ax.spines.values():
                    spine.set_visible(False)
                panels.append({"family": family, "label": label, "title_color": "black",
                               "source_attempt": row["source_attempt"], "sites": list(SITES),
                               "counts": row["layers"], "sparsity_percent": cells})
            for ax in axes[1]:
                ax.set_xlabel("Layer", labelpad=6)
            for ax in axes[:, 0]:
                ax.set_ylabel("Activation site", labelpad=8)
            cax = fig.add_axes([.918, .16, .015, .63])
            bar = fig.colorbar(im, cax=cax, ticks=[0, 20, 40, 60, 80, 100])
            bar.set_label("Activation sparsity (%)", labelpad=9)
            bar.outline.set_visible(False)
            bar.ax.tick_params(length=3, width=.5)
            pdf.savefig(fig)
            plt.close(fig)
            pages.append({"kappa": kappa, "panels": panels})
    assert sum(np.asarray(p["sparsity_percent"]).size for page in pages for p in page["panels"]) == 504
    shutil.copyfile(OUTPUT, REVIEW)
    assert all(sha(p) == digest for p, digest in preserved.items())
    result = {"source": SOURCE.relative_to(ROOT).as_posix(), "source_sha256": sha(SOURCE),
              "output": OUTPUT.relative_to(ROOT).as_posix(), "review_copy": REVIEW.relative_to(ROOT).as_posix(),
              "pdf_sha256": sha(OUTPUT), "coverage": data["coverage"], "colormap": "viridis",
              "cell_text_color": "black", "cell_text_outline": "thin translucent white",
              "color_limits_percent": [0, 100], "pages": pages}
    (HERE / "data/layer-sparsity-review.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Verified 504 cells; saved two-page PDF: {REVIEW.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

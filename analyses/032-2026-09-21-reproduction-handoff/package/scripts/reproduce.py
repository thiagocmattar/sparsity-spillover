"""Offline reconstruction from the paper's compact, immutable measurements."""

import argparse
import collections
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
OUT = ROOT / "reproduced"


def read(name):
    return json.loads((ROOT / "results" / f"{name}.json").read_text())


def save(name, data):
    OUT.mkdir(exist_ok=True)
    (OUT / name).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def verify():
    rows = json.loads((ROOT / "MANIFEST.json").read_text())["files"]
    for row in rows:
        path = (ROOT / row["path"]).resolve()
        if not path.is_relative_to(ROOT):
            raise ValueError("Unsafe manifest path")
        data = path.read_bytes()
        if (
            len(data) != row["bytes"]
            or hashlib.sha256(data).hexdigest() != row["sha256"]
        ):
            raise ValueError(f"Changed release file: {row['path']}")
    print(f"Verified {len(rows)} release files")


def results():
    from training.config import identifier, resolve, register_hz
    from sparsity_research.ceilings import architecture_ceiling

    register_hz()
    data = read("endpoints")
    rows = data["trained_points"]
    assert collections.Counter(r["model"] for r in rows) == {
        "14M": 45,
        "31M": 11,
        "70M": 27,
        "410M": 12,
    }
    assert len({identifier(r) for r in rows}) == 95
    for r in rows:
        assert math.isclose(
            100 * r["zero_product_count"] / r["model_product_count"],
            r["sparsity"],
            abs_tol=1e-10,
        )
        assert (
            r["coverage"]["sequences"] == 338
            and r["coverage"]["excluded_tail_tokens"] == 1444
        )
        resolve(identifier(r))
    ceilings = {}
    for size, l, d in [("14M", 6, 128), ("31M", 6, 256), ("70M", 6, 512), ("410M", 24, 1024)]:
        for topology in ("A0", "A1-H", "HZ", "A4-Z", "A7-Z-POST"):
            ceilings[size + "-" + topology] = architecture_ceiling(
                topology,
                layers=l,
                hidden_size=d,
                ffn_size=4 * d,
                sequence_length=2048,
                vocabulary_size=50304,
            )
    op = read("operation-latency")
    for r in op["rows"]:
        assert (
            r["cross_process_difference_span_us"][0]
            <= r["saved_microseconds"]
            <= r["cross_process_difference_span_us"][1]
        )
    controls = read("70m-controls")["rows"]
    sparse = {
        r["implementation"]: r["candidate_ms"]
        for r in controls
        if r["condition"] == "70M-T7-Ph-0.5"
    }
    benefit = 100 * (1 - sparse["specialized-70m"] / sparse["native-hz"])
    assert round(benefit, 1) == 17.4
    from scripts.verify_scale import verify_scale
    verify_scale()
    OUT.mkdir(exist_ok=True)
    fields = [
        "condition",
        "model",
        "loss",
        "sparsity_percent",
        "latency_ms",
        "timing_session",
        "zero_product_count",
        "model_product_count",
        "checkpoint_key",
    ]
    with (OUT / "endpoints.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fields)
        w.writeheader()
        for r in rows:
            w.writerow(
                {
                    k: identifier(r)
                    if k == "condition"
                    else r["sparsity"]
                    if k == "sparsity_percent"
                    else r.get(k)
                    for k in fields
                }
            )
    save(
        "checks.json",
        dict(
            endpoints=95,
            coverage=dict(validation_documents=500, blocks=338, tail=1444),
            ceilings=ceilings,
            conditional_70m_hz_reduction_percent=benefit,
            operation_effects=op["rows"],
            timing_sessions_note=data["timing_note"],
        ),
    )
    print(
        "Reconstructed 95 endpoints, 20 analytic ceilings, and central execution effects"
    )


def figures():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.backends.backend_pdf import PdfPages

    plt.rcParams.update(
        {
            "pdf.fonttype": 42,
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    OUT.mkdir(exist_ok=True)
    endpoint = read("endpoints")
    rows = endpoint["trained_points"]
    small = read("14m-figure")
    styles = {(s["scope"], s["pressure"]): s for s in small["series"]}

    def label(key):
        return styles[key]["label"] if key in styles else str(key)

    def groups(ax, items, y, only_shared=False, x="sparsity"):
        for key in dict.fromkeys((r["scope"], r["pressure"]) for r in items):
            if only_shared and key not in [
                ("0", "none"),
                ("1", "none"),
                ("hz", "h"),
                ("4", "h"),
                ("4", "all"),
                ("7", "h"),
                ("7", "all"),
            ]:
                continue
            group = sorted(
                [r for r in items if (r["scope"], r["pressure"]) == key],
                key=lambda r: r.get("kappa") or r.get("local_pressure_weight") or 0,
            )
            group = [r for r in group if r.get(y) is not None]
            if not group:
                continue
            s = styles.get(key, {})
            ls = s.get("linestyle", "-")
            ls = tuple(ls) if isinstance(ls, list) else ls
            ax.plot(
                [r[x] for r in group],
                [r[y] for r in group],
                marker="o",
                ms=3,
                lw=1,
                label=label(key),
                color=s.get("color"),
                ls=ls,
            )
        ax.set_xlabel("Validation loss" if x == "loss" else "Model-wide logical sparsity (%)")
        ax.set_ylabel("Validation loss" if y == "loss" else "Full-model latency (ms)")
        ax.grid(alpha=0.15)

    def finish(fig, name, bottom=0):
        fig.tight_layout(rect=(0, bottom, 1, 1))
        fig.savefig(OUT / name, metadata={"CreationDate": None, "ModDate": None})
        plt.close(fig)

    for size in ("14M", "70M", "410M"):
        fig, axes = plt.subplots(
            1, 1 if size == "410M" else 2, figsize=(10, 4), squeeze=False
        )
        points = (
            small["trained_points"]
            if size == "14M"
            else [r for r in rows if r["model"] == size]
        )
        groups(axes[0, 0], points, "loss")
        clips = [r for r in endpoint["clipping_points"] if r["model"] == size]
        for scope in ("0", "1"):
            path = sorted(
                [r for r in clips if r["scope"] == scope], key=lambda r: r["target"]
            )
            axes[0, 0].plot(
                [r["sparsity"] for r in path],
                [r["loss"] for r in path],
                ":",
                color=styles[(scope, "none")]["color"],
            )
        axes[0, 0].set_title(size + " quality")
        axes[0, 0].legend(fontsize=6)
        if size != "410M":
            groups(
                axes[0, 1],
                points,
                "displayed_latency_ms" if size == "70M" else "latency_ms",
            )
            ref = (
                endpoint["matched_summary"]["native_base_ms"]
                if size == "70M"
                else 0.65544
            )
            axes[0, 1].axhline(ref, ls=":", color="black", label="Native Base")
            axes[0, 1].set_title(size + " latency")
        finish(fig, size.lower() + "-quality-latency.pdf")
    op = read("operation-latency")["rows"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    current = read("14m-paper-figure")["points"]
    groups(axes[0], current, "latency_ms")
    groups(axes[2], current, "latency_ms", x="loss")
    ys = [r["saved_microseconds"] for r in op]
    errors = [
        [y - r["cross_process_difference_span_us"][0] for y, r in zip(ys, op)],
        [r["cross_process_difference_span_us"][1] - y for y, r in zip(ys, op)],
    ]
    axes[1].bar([r["operation"] for r in op], ys, yerr=errors, capsize=3)
    axes[1].set_ylabel("Conditional saved time (μs)")
    axes[1].axhline(0, color="gray", lw=0.5)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False, fontsize=7)
    finish(fig, "quality-savings-sparsity.pdf", bottom=.15)
    layer = read("pressure-placement")["rows"]
    sites = ["a", "m", "h", "q_post", "k_post", "v", "z"]
    families = ["A4", "A4-OL1-H", "A4-OL1", "A7", "A7-OL1-H", "A7-OL1"]
    with PdfPages(OUT / "layer-sparsity.pdf") as pdf:
        for k in (0.05, 0.5):
            fig, axs = plt.subplots(2, 3, figsize=(11, 6), layout="constrained")
            for ax, family in zip(axs.flat, families):
                row = next(
                    r for r in layer if r["family"] == family and r["kappa"] == k
                )
                values = np.array(
                    [
                        [100 * c["zero"] / c["total"] for c in row["layers"][s]]
                        for s in sites
                    ]
                )
                im = ax.imshow(values, vmin=0, vmax=100, aspect="auto")
                ax.set_yticks(range(7), sites)
                ax.set_xticks(range(6))
                ax.set_title(family)
            fig.colorbar(im, ax=list(axs.flat), label="Exact-zero activation (%)")
            fig.suptitle(f"14M κ={k:g}")
            pdf.savefig(fig)
            plt.close(fig)
    geometry = read("ol1-geometry")["geometry"]["conditions"]
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    for key in [("4", "h"), ("4", "all"), ("7", "h"), ("7", "all")]:
        rr = sorted(
            [
                r
                for r in geometry
                if r["model"] == "14M" and (r["scope"], r["pressure"]) == key
            ],
            key=lambda r: r["kappa"],
        )
        axs[0].plot(
            [r["kappa"] for r in rr],
            [100 * r["rho_opp"]["median"] for r in rr],
            "o-",
            label=label(key),
        )
        axs[1].plot(
            np.arange(1, 713),
            np.median([r["trace"]["r_over_b"] for r in rr], axis=0),
            label=label(key),
        )
    for ax in axs:
        ax.set_yscale("log")
        ax.legend(fontsize=7)
    axs[0].set(xlabel="κ", ylabel="Median removed opposing component (%)")
    axs[1].set(xlabel="Update", ylabel="Median pre-cap ratio r/b")
    axs[1].axhline(1, color="black", ls=":")
    finish(fig, "ol1-geometry.pdf")
    base = read("base-training")["runs"]
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    for r in base:
        for ax, key in zip(
            axs, ["smoothed_task_loss", "smoothed_gradient_norm_pre_clip"]
        ):
            ax.plot(range(1, 713), r[key], label=r["scale"])
    axs[0].set_ylabel("Training loss")
    axs[1].set_ylabel("Gradient norm before clipping")
    axs[1].set_yscale("log")
    for ax in axs:
        ax.set_xlabel("Update")
        ax.legend()
    finish(fig, "base-training.pdf")
    points = [r for r in read("kernel-structure")["points"] if r["model"] == "14M"]
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    for ax, x, y in [
        (
            axs[0],
            lambda r: r["projection"]["scalar_zero_percent"],
            lambda r: r["projection"]["bypass_percent"],
        ),
        (
            axs[1],
            lambda r: r["projection"]["bypass_percent"],
            lambda r: r["native_base_speedup"],
        ),
        (
            axs[2],
            lambda r: r["attention"]["bypass_percent"],
            lambda r: r["native_base_speedup"],
        ),
    ]:
        for kind in ("trained", "clipping"):
            group = [r for r in points if r["kind"] == kind]
            ax.scatter([x(r) for r in group], [y(r) for r in group], s=14, label=kind)
    axs[0].set(
        xlabel="Projection scalar sparsity (%)", ylabel="Projection MMA bypass (%)"
    )
    for ax, name in zip(axs[1:], ["Projection", "Attention"]):
        ax.set(xlabel=name + " MMA bypass (%)", ylabel="Native Base speedup")
        ax.axhline(1, ls=":", color="gray")
    finish(fig, "kernel-structure.pdf")
    clip = read("clipping")["posthoc_points"]
    fig, ax = plt.subplots(figsize=(7, 4))
    for source in dict.fromkeys(r["condition"] for r in clip):
        group = sorted(
            [r for r in clip if r["condition"] == source],
            key=lambda r: r["target"],
        )
        ax.plot(
            [r["sparsity"] for r in group],
            [r["loss"] for r in group],
            ":o",
            ms=2,
            lw=0.8,
        )
    ax.set(
        xlabel="Model-wide logical sparsity (%)",
        ylabel="Validation loss",
        title="14M post-hoc clipping: 30 × 10 settings",
    )
    finish(fig, "posthoc-clipping.pdf")
    from scripts.plot_scale import main as plot_scale
    plot_scale()
    print("Wrote central figure reconstructions to reproduced/")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["verify", "results", "figures"])
    a = p.parse_args()
    globals()[a.action]()

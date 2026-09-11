"""A0 loss and pre-clipping global gradient norms from complete training logs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
COHORT = ROOT / "analyses/018-2026-09-08-results-materials/figure_data.json"
COLORS = {"14M": "#2878B5", "70M": "#C96024", "410M": "#7563A8"}
WINDOW = 9
FIELDS = ("step", "input_tokens_seen", "task_loss", "adamw_gradient_norm_pre_clip",
          "adamw_gradient_norm_post_clip", "adamw_gradient_clip_norm",
          "adamw_gradient_was_clipped", "learning_rate", "optimizer_step_skipped", "gradient_overflow")


def smooth(values, window=WINDOW):
    """Centered arithmetic mean; shorten the window at edges, without padding."""
    values = np.asarray(values, dtype=float)
    assert window % 2 == 1 and 1 <= window <= len(values)
    weights = np.ones(window)
    return np.convolve(values, weights, mode="same") / np.convolve(np.ones(len(values)), weights, mode="same")


def read_evidence():
    cohort = json.loads(COHORT.read_text(encoding="utf-8"))
    hashes = {COHORT.relative_to(ROOT).as_posix(): hashlib.sha256(COHORT.read_bytes()).hexdigest()}
    runs = []
    for size in COLORS:
        baseline, = [r for r in cohort["trained"] if r["scale"] == size and r["family"] == "A0"]
        attempt = ROOT / baseline["source"]
        events_path, manifest_path = attempt / "events.jsonl", attempt / "manifest.json"
        for path in (events_path, manifest_path):
            hashes[path.relative_to(ROOT).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        events = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines()]
        events = [e for e in events if e["event"] == "train"]
        assert len({e["condition_id"] for e in events}) == 1
        assert len(events) == manifest["completed_steps"] == 712
        assert [e["step"] for e in events] == list(range(1, 713))
        identity = baseline["identity"]
        assert manifest["initial_parameter_sha256"] == identity["initial_parameter_sha256"]
        assert manifest["training_schedule_hash"] == identity["schedule_sha256"]
        assert identity["pressure"]["method"] == "none"
        tokens_per_step = identity["training_tokens"] // len(events)
        assert [e["input_tokens_seen"] for e in events] == [tokens_per_step * i for i in range(1, 713)]
        assert events[-1]["input_tokens_seen"] == manifest["input_tokens"] == identity["training_tokens"]
        assert all(not e["optimizer_step_skipped"] and not e["gradient_overflow"] for e in events)
        for e in events:
            assert np.isfinite(e["task_loss"]) and e["task_loss"] > 0
            assert np.isfinite(e["adamw_gradient_norm_pre_clip"]) and e["adamw_gradient_norm_pre_clip"] > 0
            assert e["adamw_gradient_clip_norm"] == 1 and e["adamw_gradient_clipping_enabled"]
            assert np.isclose(e["adamw_gradient_norm_post_clip"], min(e["adamw_gradient_norm_pre_clip"], 1), atol=2e-6)
        rows = [{k: e[k] for k in FIELDS} for e in events]
        runs.append({
            "scale": size, "baseline_id": baseline["id"], "condition_id": events[0]["condition_id"],
            "events_source": events_path.relative_to(ROOT).as_posix(),
            "manifest_source": manifest_path.relative_to(ROOT).as_posix(),
            "identity": identity, "code_identity": manifest["code"],
            "tokens_per_step": tokens_per_step, "rows": rows,
            "training_tokens_billions": [e["input_tokens_seen"] / 1e9 for e in rows],
            "smoothed_task_loss": smooth([e["task_loss"] for e in rows]).tolist(),
            "smoothed_gradient_norm_pre_clip": smooth([e["adamw_gradient_norm_pre_clip"] for e in rows]).tolist(),
            "summary": {
                "final_training_loss": rows[-1]["task_loss"],
                "final_pre_clip_norm": rows[-1]["adamw_gradient_norm_pre_clip"],
                "maximum_pre_clip_norm": max(e["adamw_gradient_norm_pre_clip"] for e in rows),
                "clipped_steps": sum(e["adamw_gradient_was_clipped"] for e in rows),
            },
        })
    return {
        "question": "How do untreated A0 training loss and pre-clipping global gradient norms evolve with the same token budget across sizes?",
        "source_sha256": hashes,
        "coverage": {"model_sizes": 3, "steps_per_size": 712, "total_step_records": 2136,
                     "tokens_per_size": 1493172224, "seed_per_size": 1234},
        "task_loss_definition": "Arithmetic mean causal-language-model task loss over equally sized microbatches at each accumulated optimizer boundary",
        "gradient_definition": "Global L2 norm over trainable-parameter task gradients, after loss unscaling and accumulation, immediately before clipping at 1.0 and before AdamW",
        "smoothing": {"window_steps": WINDOW, "window_tokens": WINDOW * runs[0]["tokens_per_step"],
                      "method": "Centered arithmetic moving mean in original units; partial windows at edges; raw traces retained and displayed"},
        "runs": runs,
    }


def make_figure(data):
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8,
        "axes.labelsize": 9, "axes.titlesize": 8,
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": .6, "pdf.fonttype": 42,
    })
    fig, axes = plt.subplots(1, 2, sharex=True, figsize=(7.4, 2.95))
    fig.subplots_adjust(left=.085, right=.985, bottom=.25, top=.87, wspace=.25)
    for ax, raw_key, smooth_key, title, ylabel in (
        (axes[0], "task_loss", "smoothed_task_loss", "(a) A0 training loss", "Training loss"),
        (axes[1], "adamw_gradient_norm_pre_clip", "smoothed_gradient_norm_pre_clip",
         "(b) A0 gradient norm", "Pre-clipping global\ntask-gradient norm"),
    ):
        ax.set(xlim=(0, 1.5), xlabel="Training tokens (B)", ylabel=ylabel)
        ax.set_title(title, loc="left", pad=9)
        ax.set_xticks([0, .5, 1, 1.5])
        ax.tick_params(length=3, width=.6)
        ax.set_axisbelow(True)
        ax.grid(axis="y", which="major", color="#ECEDEF", linewidth=.45)
        for run in data["runs"]:
            color = COLORS[run["scale"]]
            ax.plot(run["training_tokens_billions"], [e[raw_key] for e in run["rows"]],
                    color=color, alpha=.12, linewidth=.4, gid=f'{run["scale"]}:{raw_key}:raw')
            ax.plot(run["training_tokens_billions"], run[smooth_key],
                    color=color, linewidth=1.2, label=run["scale"], gid=f'{run["scale"]}:{raw_key}:smooth')
    axes[0].set_ylim(3.5, 11.5)
    axes[0].set_yticks([4, 6, 8, 10])
    axes[1].set(yscale="log", ylim=(.15, 35))
    axes[1].tick_params(axis="y", which="minor", length=0)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="center", bbox_to_anchor=(.54, .045),
               ncol=3, frameon=False, fontsize=8, handlelength=2.3, columnspacing=2.5)
    return fig


def main():
    data = read_evidence()
    (HERE / "data/a0-optimization.json").write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8", newline="\n")
    fig = make_figure(data)
    path = HERE / "figures/06-a0-optimization.pdf"
    fig.savefig(path, metadata={"Title": "A0 optimization across Pythia model sizes",
                              "Subject": "Analysis 021 O007; complete task-loss and pre-clipping gradient histories",
                              "CreationDate": None, "ModDate": None})
    plt.close(fig)
    print(path.relative_to(ROOT).as_posix())
    for run in data["runs"]:
        print(run["scale"], run["summary"])


if __name__ == "__main__":
    main()

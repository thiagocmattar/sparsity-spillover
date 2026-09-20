"""Use the manuscript's T/P labels on the unchanged base-model trajectories."""

import hashlib
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ORIGINAL = ROOT / "analyses/021-2026-09-10-training-results-figures"
SOURCE = ORIGINAL / "06_a0_optimization.py"
DATA = ORIGINAL / "data/a0-optimization.json"
OUTPUT = HERE / "figures/19-base-model-optimization.pdf"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    spec = importlib.util.spec_from_file_location("base_optimization", SOURCE)
    original = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(original)
    data = json.loads(DATA.read_text(encoding="utf-8"))
    assert original.read_evidence() == data
    figure = original.make_figure(data)
    before = [[(line.get_xdata().tolist(), line.get_ydata().tolist()) for line in ax.lines]
              for ax in figure.axes]
    for ax, title in zip(figure.axes, (r"(a) $T_0/P_0$ training loss", r"(b) $T_0/P_0$ gradient norm")):
        ax.set_title(title, loc="left", pad=9)
    after = [[(line.get_xdata().tolist(), line.get_ydata().tolist()) for line in ax.lines]
             for ax in figure.axes]
    assert before == after
    assert len(figure.axes) == 2 and all(len(ax.lines) == 6 for ax in figure.axes)
    figure.savefig(OUTPUT, metadata={"Title": "Base-model (T0/P0) optimization across model sizes",
                                   "CreationDate": None, "ModDate": None})
    original.plt.close(figure)
    provenance = {"source_script": SOURCE.relative_to(ROOT).as_posix(),
                  "source_script_sha256": sha(SOURCE),
                  "source_data": DATA.relative_to(ROOT).as_posix(), "source_data_sha256": sha(DATA),
                  "builder_sha256": sha(Path(__file__)), "coverage": data["coverage"],
                  "change": "Only panel titles and PDF metadata; all raw and smoothed coordinates unchanged",
                  "output": OUTPUT.relative_to(ROOT).as_posix(), "output_sha256": sha(OUTPUT)}
    (HERE / "data/base-model-optimization-labels.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"Verified {data['coverage']['total_step_records']} records; changed only labels: {OUTPUT.name}")


if __name__ == "__main__":
    main()

"""Figure 07 v2: model-wide sparsity against both full and projection gains."""

import hashlib
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("kernel_figure_base", HERE / "07_kernel_realization.py")
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)


def read_evidence():
    data = base.read_evidence()
    full = {p["condition"]: p for p in data["points"] if p["candidate"] == "k050"}
    for point in data["projection_points"]:
        checkpoint = full[point["condition"]]
        assert checkpoint["evidence_id"] == point["evidence_id"]
        point["sparsity_percent"] = checkpoint["sparsity_percent"]
    data["projection_regression"] = base.regression(data["projection_points"], "sparsity_percent", "projection_sparse_gain")
    data["projection_x_metric"] = "sparsity_percent"
    data["question"] = "How does model-wide sparsity relate to native-relative full K050 speedup and the matched gain from projection skipping?"
    data["variant"] = "v2; model-wide sparsity replaces projection MMA bypass on panel (b)"
    original = HERE / "figures/07-kernel-realization.pdf"
    data["original_figure_sha256"] = hashlib.sha256(original.read_bytes()).hexdigest()
    return data


def main():
    data = read_evidence()
    (HERE / "data/kernel-realization-v2.json").write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    fig = base.make_figure(data, projection_x=data["projection_x_metric"])
    fig.savefig(HERE / "figures/07-kernel-realization-v2.pdf",
                metadata={"Title": "Model-wide sparsity and K050 full-model and projection-skipping gains",
                          "Creator": "Analysis 021 / 07_kernel_realization_v2.py", "CreationDate": None})
    base.plt.close(fig)
    assert hashlib.sha256((HERE / "figures/07-kernel-realization.pdf").read_bytes()).hexdigest() == data["original_figure_sha256"]
    print(f'30 checkpoints; panel (a) R²={data["regression"]["r2"]:.6f}; panel (b) R²={data["projection_regression"]["r2"]:.6f}')


if __name__ == "__main__":
    main()

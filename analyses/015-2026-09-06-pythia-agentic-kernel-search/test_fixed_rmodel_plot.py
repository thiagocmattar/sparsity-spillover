import importlib.util
from pathlib import Path


SOURCE = Path(__file__).with_name("06_plot_fixed_rmodel.py")
SPEC = importlib.util.spec_from_file_location("run025_fixed_plot", SOURCE)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_plot_renders_three_architectures(tmp_path: Path) -> None:
    pairs = []
    summaries = []
    transitions = {"14M": ("p0", "k001"), "70M": ("k009", "k016"), "410M": ("k004", "k010")}
    for size_index, size in enumerate(MODULE.SIZE_ORDER):
        baseline, optimized = transitions[size]
        for family_index, family in enumerate(("A4-OL1", "A7-OL1")):
            before = 0.98 + 0.02 * size_index + 0.005 * family_index
            after = before + 0.025
            for repeat in (1, 2, 3):
                pairs.append({
                    "model_size": size,
                    "family": family,
                    "repeat": str(repeat),
                    "baseline_speedup": str(before + 0.001 * repeat),
                    "optimized_speedup": str(after + 0.001 * repeat),
                })
            summaries.append({
                "model_size": size,
                "family": family,
                "baseline_implementation": baseline,
                "optimized_implementation": optimized,
                "baseline_median_speedup": str(before + 0.002),
                "optimized_median_speedup": str(after + 0.002),
                "median_optimized_over_baseline": str((after + 0.002) / (before + 0.002)),
                "R_model_percent": str(10 + 15 * size_index + 5 * family_index),
            })
    output = tmp_path / "fixed.pdf"
    MODULE.plot(pairs, summaries, output)
    assert output.read_bytes().startswith(b"%PDF")
    assert output.stat().st_size > 10_000

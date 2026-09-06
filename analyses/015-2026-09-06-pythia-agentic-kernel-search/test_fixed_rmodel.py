import importlib.util
from pathlib import Path


SOURCE = Path(__file__).with_name("05_reduce_fixed_rmodel.py")
SPEC = importlib.util.spec_from_file_location("run025_fixed_rmodel", SOURCE)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def synthetic_rows() -> list[dict]:
    rows = []
    for condition, (_, _, _, baseline, optimized) in MODULE.PAIR_SPECS.items():
        rmodel = {"14m": 0.40, "70m": 0.50, "410m": 0.75}[condition.split("/", 1)[0]]
        for repeat in (1, 2, 3):
            rows += [
                {
                    "label": f"fixed-r{repeat}-condition-{baseline}-rtxpro4500-004",
                    "condition": condition,
                    "implementation": baseline,
                    "candidate_validation_pass": True,
                    "R_model": rmodel,
                    "paired_geomean_speedup": 0.9 + 0.01 * repeat,
                },
                {
                    "label": f"fixed-r{repeat}-condition-{optimized}-rtxpro4500-004",
                    "condition": condition,
                    "implementation": optimized,
                    "candidate_validation_pass": True,
                    "R_model": rmodel,
                    "paired_geomean_speedup": 1.0 + 0.01 * repeat,
                },
            ]
    return rows


def test_reduction_accounts_exact_design_and_fixed_rmodel() -> None:
    reduced = MODULE.reduce_rows(synthetic_rows())
    assert len(reduced["pair_rows"]) == 18
    assert len(reduced["condition_rows"]) == 6
    assert len(reduced["architecture_rows"]) == 3
    assert all(row["qualified_repeats"] == 3 for row in reduced["condition_rows"])
    assert all(row["optimized_faster_repeats"] == 3 for row in reduced["condition_rows"])


def test_reduction_rejects_changed_rmodel() -> None:
    rows = synthetic_rows()
    rows[0]["R_model"] += 0.01
    try:
        MODULE.reduce_rows(rows)
    except ValueError as error:
        assert "R_model differs" in str(error)
    else:
        raise AssertionError("changed R_model was accepted")


def test_reduction_rejects_missing_process() -> None:
    try:
        MODULE.reduce_rows(synthetic_rows()[:-1])
    except ValueError as error:
        assert "Expected 36 process rows" in str(error)
    else:
        raise AssertionError("missing process was accepted")


def test_phase18_reduction_uses_k001_as_optimized_kernel() -> None:
    rows = [row for row in synthetic_rows() if row["condition"] in MODULE.PHASE18_SPECS]
    for row in rows:
        if row["implementation"] == "k013":
            row["implementation"] = "k001"
            row["label"] = row["label"].replace("k013", "k001")
        row["label"] = "k001" + row["label"]
    reduced = MODULE.reduce_rows(rows, MODULE.PHASE18_SPECS, MODULE.PHASE18_REPEAT)
    assert len(reduced["condition_rows"]) == 2
    assert reduced["architecture_rows"][0]["model_size"] == "14M"
    assert all(row["optimized_implementation"] == "k001" for row in reduced["condition_rows"])

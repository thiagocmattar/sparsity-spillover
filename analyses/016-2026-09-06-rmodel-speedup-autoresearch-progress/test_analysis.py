from __future__ import annotations

import csv
import importlib.util
import math
from pathlib import Path


HERE = Path(__file__).resolve().parent


def load_reducer():
    spec = importlib.util.spec_from_file_location("analysis016_reduce", HERE / "01_reduce.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def rows(path: str) -> list[dict[str, str]]:
    with (HERE / path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def test_reduction_contract() -> None:
    reducer = load_reducer()
    reducer.main()
    rmodel = rows("rmodel-speedup.csv")
    fits = rows("rmodel-regressions.csv")
    progress = rows("candidate-progress.csv")
    assert len(rmodel) == 54
    assert len(fits) == 6
    assert len(progress) == 20
    assert sum(row["qualified"] == "True" for row in rmodel) == 50
    assert {(row["gpu"], row["model_size"]) for row in fits} == {
        (gpu, size) for gpu in reducer.GPU_ORDER for size in reducer.SIZE_ORDER
    }


def test_progress_scores_and_statuses() -> None:
    progress = rows("candidate-progress.csv")
    scored = [row for row in progress if row["endpoint_geomean_speedup"]]
    assert len(scored) == 12
    for row in scored:
        expected = math.sqrt(float(row["a4_high_speedup"]) * float(row["a7_high_speedup"]))
        assert math.isclose(expected, float(row["endpoint_geomean_speedup"]), rel_tol=1e-14)
    keyed = {(row["model_size"], row["candidate_id"]): row for row in progress}
    assert keyed[("14M", "K012")]["status"] == "rejected_full_matrix"
    assert keyed[("14M", "K013")]["status"] == "frozen_final"
    assert keyed[("70M", "K016")]["status"] == "frozen_final"
    assert keyed[("410M", "K010")]["status"] == "frozen_final"
    assert keyed[("410M", "P0")]["status"] == "gate_failure"
    assert float(keyed[("410M", "K004")]["endpoint_geomean_speedup"]) < 0.60
    assert float(keyed[("410M", "K010")]["endpoint_geomean_speedup"]) > 1.01


def test_integer_pooled_rmodel_is_retained() -> None:
    for row in rows("rmodel-speedup.csv"):
        expected = int(row["block_zero_product_count"]) / int(row["model_product_count"])
        assert math.isclose(expected, float(row["measured_R_model_fraction"]), abs_tol=5e-16)

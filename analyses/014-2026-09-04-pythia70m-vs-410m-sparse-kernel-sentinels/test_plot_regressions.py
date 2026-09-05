from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import numpy as np
import pytest


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("rmodel_speedup_plot", HERE / "02_plot_rmodel_speedup.py")
PLOT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PLOT)


def test_regression_recovers_intercept_and_nontrivial_r_squared() -> None:
    # y = 2 + 3x + [1, -2, 1]; residual is orthogonal to both 1 and x.
    points = [
        {"model_size": size, "batch_size": batch, "R_model": x,
         "speedup_median": y, "timing_blocks": 100 if x == 0 else 1}
        for size in PLOT.RUNS for batch in (1, 32)
        for x, y in zip((0, 1, 2), (3, 3, 9))
    ]
    fits = PLOT.fit_regressions(points)
    assert len(fits) == 4
    for fit in fits:
        assert fit["n_checkpoints"] == 3
        assert fit["intercept"] == pytest.approx(2)
        assert fit["slope"] == pytest.approx(3)
        assert fit["R_squared"] == pytest.approx(0.75)


def test_exported_fits_match_independent_least_squares_per_size_and_batch() -> None:
    with (HERE / "rmodel-speedup-points.csv").open(newline="", encoding="utf-8") as stream:
        points = list(csv.DictReader(stream))
    with (HERE / "rmodel-speedup-regressions.csv").open(newline="", encoding="utf-8") as stream:
        fits = list(csv.DictReader(stream))
    assert {(fit["model_size"], fit["batch_size"]) for fit in fits} == {
        (size, batch) for size in PLOT.RUNS for batch in ("1", "32")
    }
    assert len(fits) == 4
    for fit in fits:
        group = [p for p in points if (p["model_size"], p["batch_size"]) ==
                 (fit["model_size"], fit["batch_size"])]
        assert len(group) == int(fit["n_checkpoints"]) == 6
        x = np.array([float(p["R_model"]) for p in group])
        y = np.array([float(p["speedup_median"]) for p in group])
        slope, intercept = np.linalg.lstsq(np.column_stack((x, np.ones_like(x))), y, rcond=None)[0]
        assert float(fit["slope"]) == pytest.approx(slope, abs=1e-12)
        assert float(fit["intercept"]) == pytest.approx(intercept, abs=1e-12)
        assert float(fit["R_squared"]) == pytest.approx(np.corrcoef(x, y)[0, 1] ** 2, abs=1e-12)
        assert float(fit["R_model_min"]) == x.min()
        assert float(fit["R_model_max"]) == x.max()

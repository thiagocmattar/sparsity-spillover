import pytest

from tools.estimate_etc import estimate


def test_etc_includes_all_phases_and_cost_ceiling():
    result = estimate(
        {
            "optimizer_step_seconds": [2.0, 2.5, 3.0],
            "full_validation_seconds": [10.0, 12.0],
            "setup_seconds": 20.0,
            "diagnostics_seconds": 5.0,
            "checkpoint_seconds": 3.0,
            "transfer_seconds": 2.0,
        },
        steps=100,
        validation_passes=3,
        gpu_count=1,
        hourly_gpu_price=0.50,
        maximum_hours=2.0,
        storage_cost=0.10,
    )
    assert result["fixed_seconds"] == 30.0
    assert result["median_etc_seconds"] == 313.0
    assert result["p90_etc_seconds"] == 366.0
    assert result["maximum_total_cost"] == pytest.approx(1.10)

import json

import pytest

import verification_observed_bound as repair


def _legacy_ceiling_failure(*_args, **_kwargs):
    raise ValueError("Measured R_model exceeds its declared reach ceiling for synthetic.")


@pytest.mark.parametrize("observed", [0.0, 4.5409213981255505e-06, 1.0])
def test_repair_accepts_valid_observed_fraction_after_legacy_ceiling_failure(
    monkeypatch, tmp_path, observed
):
    diagnostics = tmp_path / "diagnostics"
    diagnostics.mkdir()
    (diagnostics / "logical_products.json").write_text(
        json.dumps({"measured": {"R_model": observed}}), encoding="utf-8"
    )
    monkeypatch.setattr(repair, "_ORIGINAL_REQUIRE_DIAGNOSTICS", _legacy_ceiling_failure)
    repair._require_diagnostics_with_observed_bound(
        tmp_path,
        {"diagnostics": {"logical_products_path": "diagnostics/logical_products.json"}},
        {},
        {},
    )


def test_repair_rejects_observed_fraction_above_one(monkeypatch, tmp_path):
    diagnostics = tmp_path / "diagnostics"
    diagnostics.mkdir()
    (diagnostics / "logical_products.json").write_text(
        json.dumps({"measured": {"R_model": 1.000001}}), encoding="utf-8"
    )
    monkeypatch.setattr(repair, "_ORIGINAL_REQUIRE_DIAGNOSTICS", _legacy_ceiling_failure)
    with pytest.raises(ValueError, match="not a valid fraction"):
        repair._require_diagnostics_with_observed_bound(
            tmp_path,
            {"diagnostics": {"logical_products_path": "diagnostics/logical_products.json"}},
            {},
            {},
        )


def test_repair_preserves_unrelated_verifier_failures(monkeypatch, tmp_path):
    def fail(*_args, **_kwargs):
        raise ValueError("activation counts do not reconcile")

    monkeypatch.setattr(repair, "_ORIGINAL_REQUIRE_DIAGNOSTICS", fail)
    with pytest.raises(ValueError, match="activation counts do not reconcile"):
        repair._require_diagnostics_with_observed_bound(tmp_path, {}, {}, {})

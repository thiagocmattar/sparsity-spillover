"""Fail-closed pressure capture driven by each resolved Run 016 condition."""

from sparsity_research.capture import ActivationCapture

from run_config import A4_SITES, A7_SITES


class ConditionPressureCapture(ActivationCapture):
    """Replace Run 004's frozen one-site request with the resolved pressure sites."""

    def __init__(self, model, sites, *, torch, clipping=None):
        if tuple(sites) != ("h",):
            raise RuntimeError("The frozen Run 004 capture call changed unexpectedly.")
        pressure_sites = tuple(getattr(model.config, "pressure_sites", ()))
        if pressure_sites not in {A4_SITES, A7_SITES}:
            raise RuntimeError(f"Unsupported Run 016 pressure-site identity: {pressure_sites}")
        super().__init__(model, list(pressure_sites), torch=torch, clipping=clipping)

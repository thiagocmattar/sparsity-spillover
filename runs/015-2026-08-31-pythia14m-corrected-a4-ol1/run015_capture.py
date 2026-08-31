"""Pressure-capture adapter for the corrected four-site A4-OL1 objective."""

from sparsity_research.capture import ActivationCapture

from run_config import EXPECTED_ACTIVE_SITES


class FourSitePressureCapture(ActivationCapture):
    """Replace the frozen Run 004 `h` request with all four A4 sites."""

    def __init__(self, model, sites, *, torch, clipping=None):
        if tuple(sites) != ("h",):
            raise RuntimeError("The frozen Run 004 capture call changed unexpectedly.")
        super().__init__(
            model,
            list(EXPECTED_ACTIVE_SITES),
            torch=torch,
            clipping=clipping,
        )

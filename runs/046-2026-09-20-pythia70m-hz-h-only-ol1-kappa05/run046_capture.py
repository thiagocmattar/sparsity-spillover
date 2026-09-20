"""Audited adapter from Run 004's one-site capture to Run 046's h-only pressure."""

from sparsity_research.capture import ActivationCapture

from run_config import EXPECTED_PRESSURE_SITES


class HOnlyPressureCapture(ActivationCapture):
    """Replace the frozen Run 004 `h` request with only the approved post-gate h sites."""

    def __init__(self, model, sites, *, torch, clipping=None):
        if tuple(sites) != ("h",):
            raise RuntimeError("The frozen Run 004 capture call changed unexpectedly.")
        super().__init__(
            model,
            list(EXPECTED_PRESSURE_SITES),
            torch=torch,
            clipping=clipping,
        )

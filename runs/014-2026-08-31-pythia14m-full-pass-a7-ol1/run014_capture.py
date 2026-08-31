"""Audited adapter from Run 004's one-site capture to Run 014's A7 set."""

from sparsity_research.capture import ActivationCapture

from run_config import EXPECTED_ACTIVE_SITES


class SevenSitePressureCapture(ActivationCapture):
    """Replace the frozen Run 004 `h` request with all seven approved sites."""

    def __init__(self, model, sites, *, torch, clipping=None):
        if tuple(sites) != ("h",):
            raise RuntimeError("The frozen Run 004 capture call changed unexpectedly.")
        super().__init__(
            model,
            list(EXPECTED_ACTIVE_SITES),
            torch=torch,
            clipping=clipping,
        )

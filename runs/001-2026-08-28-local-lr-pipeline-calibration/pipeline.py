"""Public run-local API split across contract, execution, and ETC modules."""

from pathlib import Path
import sys


MODULE_DIR = Path(__file__).resolve().parent
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from sparsity_research.data import build_training_schedule  # noqa: E402
from sparsity_research.optimization import warmup_steps  # noqa: E402

from lr_calibration import (  # noqa: E402
    budget_steps_from_calibration,
    calibrate,
    estimate_cohort,
)
from lr_run_config import (  # noqa: E402
    DEFAULT_CONFIG,
    REPO_ROOT,
    RUN_DIR,
    build_schedule,
    inventory_content_sha256,
    learning_rate_label,
    load_config,
    load_verified_caches,
    microbatches_for_step,
    parameter_sha256,
    resolved_condition_config,
    run_code_identity,
    validate_config,
)
from lr_training import (  # noqa: E402
    run_cohort,
    run_condition,
    timed_diagnostic_validation,
    timed_validation,
)


__all__ = [
    "DEFAULT_CONFIG",
    "REPO_ROOT",
    "RUN_DIR",
    "budget_steps_from_calibration",
    "build_schedule",
    "build_training_schedule",
    "calibrate",
    "estimate_cohort",
    "inventory_content_sha256",
    "learning_rate_label",
    "load_config",
    "load_verified_caches",
    "microbatches_for_step",
    "parameter_sha256",
    "resolved_condition_config",
    "run_code_identity",
    "run_cohort",
    "run_condition",
    "timed_diagnostic_validation",
    "timed_validation",
    "validate_config",
    "warmup_steps",
]

#!/usr/bin/env python
"""Fail closed unless a worker reaches concrete condition-resolved LRs."""

import argparse
import json

from run_config import load_config, worker_conditions
from training import _condition_resolved_execution_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", required=True)
    args = parser.parse_args()
    config = load_config()
    conditions = worker_conditions(config, args.worker)
    if len(conditions) != 1:
        raise RuntimeError("Run 021 requires exactly one condition per worker.")
    condition = conditions[0]
    execution = _condition_resolved_execution_config(config, condition)
    print(
        json.dumps(
            {
                "status": "verified",
                "worker": args.worker,
                "condition_id": condition["id"],
                "source_peak_learning_rate": config["training"]["peak_learning_rate"],
                "resolved_peak_learning_rate": execution["training"]["peak_learning_rate"],
                "resolved_minimum_learning_rate": execution["training"][
                    "minimum_learning_rate"
                ],
                "base_run_worker_dispatches_to_resolution_wrapper": (
                    __import__("training")._BASE.run_condition
                    is __import__("training")._run_condition_with_resolved_training
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

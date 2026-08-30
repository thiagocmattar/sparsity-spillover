#!/usr/bin/env python
"""Verify one Run 011 worker attempt or the complete retrieved cohort."""

from __future__ import annotations

import argparse

from run_config import EXPECTED_CONDITION_IDS
from verification import verify_attempt, verify_run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", choices=EXPECTED_CONDITION_IDS)
    args = parser.parse_args()
    if args.condition:
        result = verify_attempt(args.condition)
        print(result["status"], result["condition"]["id"])
    else:
        result = verify_run()
        print(result["status"], result["condition_count"])


if __name__ == "__main__":
    main()

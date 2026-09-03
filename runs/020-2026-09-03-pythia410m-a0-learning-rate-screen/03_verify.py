#!/usr/bin/env python
"""Verify one Run 020 attempt or the complete retrieved cohort."""

import argparse

from run_config import EXPECTED_CONDITION_IDS
from verification import verify_attempt, verify_run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", choices=EXPECTED_CONDITION_IDS)
    args = parser.parse_args()
    result = verify_attempt(args.condition) if args.condition else verify_run()
    print(result["status"], result.get("condition", {}).get("id", result.get("condition_count")))


if __name__ == "__main__":
    main()

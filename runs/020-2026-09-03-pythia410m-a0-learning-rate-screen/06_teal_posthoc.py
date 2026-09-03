#!/usr/bin/env python
"""Evaluate or consolidate the training-selected Run 020 A0 TEAL frontier."""

import argparse

from run_config import EXPECTED_CONDITION_IDS
from teal_posthoc import consolidate, evaluate_condition


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--condition", choices=EXPECTED_CONDITION_IDS)
    group.add_argument("--consolidate", action="store_true")
    args = parser.parse_args()
    output = consolidate() if args.consolidate else evaluate_condition(args.condition)
    print(output)


if __name__ == "__main__":
    main()

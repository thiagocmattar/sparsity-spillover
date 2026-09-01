#!/usr/bin/env python
"""Evaluate one Run 017 control TEAL frontier or consolidate both."""

import argparse

from teal_posthoc import consolidate, evaluate_condition


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--condition", choices=("a0-gelu", "a1h-relu"))
    group.add_argument("--consolidate", action="store_true")
    args = parser.parse_args()
    output = consolidate() if args.consolidate else evaluate_condition(args.condition)
    print(output)


if __name__ == "__main__":
    main()

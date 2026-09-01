#!/usr/bin/env python
"""Verify Run 018 attempts or the cohort with the observed-R_model repair."""

import argparse

from verification_observed_bound import verify_attempt, verify_run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition")
    args = parser.parse_args()
    if args.condition:
        result = verify_attempt(args.condition)
        print(f"verified {result['condition']['id']} with observed R_model bound")
    else:
        result = verify_run()
        print(f"verified {result['condition_count']} conditions with observed R_model bound")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Verify one Run 018 attempt with the observed-R_model bound repair."""

import argparse

from verification_observed_bound import verify_attempt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", required=True)
    args = parser.parse_args()
    result = verify_attempt(args.condition)
    print(f"verified {result['condition']['id']} with observed R_model bound")


if __name__ == "__main__":
    main()

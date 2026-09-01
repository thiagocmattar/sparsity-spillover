#!/usr/bin/env python
"""Run TEAL after applying the append-only observed-R_model verifier repair."""

import argparse

import teal_posthoc as _teal_proxy
from verification_observed_bound import verify_attempt


_FROZEN_TEAL = _teal_proxy._FROZEN


def evaluate_condition(condition_id: str):
    original = _FROZEN_TEAL.verify_attempt
    _FROZEN_TEAL.verify_attempt = verify_attempt
    try:
        return _FROZEN_TEAL.evaluate_condition(condition_id)
    finally:
        _FROZEN_TEAL.verify_attempt = original


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", choices=("a0-gelu",), required=True)
    args = parser.parse_args()
    print(evaluate_condition(args.condition))


if __name__ == "__main__":
    main()

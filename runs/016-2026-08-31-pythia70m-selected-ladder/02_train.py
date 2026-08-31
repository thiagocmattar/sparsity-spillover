#!/usr/bin/env python
"""Run one independently assigned Run 016 condition."""

import argparse

from run_config import EXPECTED_WORKERS
from training import run_worker


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", required=True, choices=tuple(EXPECTED_WORKERS))
    args = parser.parse_args()
    for path in run_worker(args.worker):
        print(path)


if __name__ == "__main__":
    main()

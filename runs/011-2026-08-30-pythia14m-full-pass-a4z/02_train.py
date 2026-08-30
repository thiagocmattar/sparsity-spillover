#!/usr/bin/env python
"""Run one preassigned Run 011 A4-Z Pod worker."""

from __future__ import annotations

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

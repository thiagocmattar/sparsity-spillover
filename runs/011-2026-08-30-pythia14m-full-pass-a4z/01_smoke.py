#!/usr/bin/env python
"""Run non-evidence Run 011 numerical and memory probes."""

from __future__ import annotations

import argparse
import json

from smoke import run_smoke


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[2, 4])
    parser.add_argument("--sequence-length", type=int, default=2048)
    parser.add_argument("--boundaries", type=int, default=2)
    parser.add_argument("--accumulation-steps", type=int, default=1)
    args = parser.parse_args()
    result = run_smoke(
        batch_sizes=args.batch_sizes,
        sequence_length=args.sequence_length,
        boundaries=args.boundaries,
        accumulation_steps=args.accumulation_steps,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Run non-evidence Run 016 structural or exact-dimension probes."""

import argparse
import json

from smoke import run_smoke


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--conditions", nargs="+")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--boundaries", type=int, default=1)
    parser.add_argument("--accumulation-steps", type=int, default=1)
    args = parser.parse_args()
    kwargs = {
        "batch_size": args.batch_size,
        "sequence_length": args.sequence_length,
        "boundaries": args.boundaries,
        "accumulation_steps": args.accumulation_steps,
    }
    if args.conditions:
        kwargs["condition_ids"] = args.conditions
    print(json.dumps(run_smoke(**kwargs), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

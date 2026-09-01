#!/usr/bin/env python
"""Execute one exact, non-evidence Run 019 GPU calibration."""

import argparse
import json
from pathlib import Path

from calibration import run_calibration


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-type-id", required=True)
    parser.add_argument("--cloud-type", required=True, choices=("COMMUNITY", "SECURE"))
    parser.add_argument("--hourly-price-usd", required=True, type=float)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = run_calibration(
        gpu_type_id=args.gpu_type_id,
        cloud_type=args.cloud_type,
        hourly_price_usd=args.hourly_price_usd,
        output=args.output,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] != "passed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()

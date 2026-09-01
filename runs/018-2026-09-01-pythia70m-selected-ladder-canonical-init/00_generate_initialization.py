#!/usr/bin/env python
"""Generate Run 018's local, untrained initialization and RNG artifacts once."""

import json

from initialization_artifact import generate_initialization_artifacts


if __name__ == "__main__":
    print(json.dumps(generate_initialization_artifacts(), indent=2, sort_keys=True))

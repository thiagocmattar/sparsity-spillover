"""Measure representative local timings without creating an evidence attempt."""

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pipeline import calibrate  # noqa: E402


if __name__ == "__main__":
    output = calibrate()
    print(output)

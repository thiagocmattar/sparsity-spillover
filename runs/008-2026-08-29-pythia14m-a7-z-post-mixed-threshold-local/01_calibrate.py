"""Run the non-evidence production-shaped local calibration for Run 008."""

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from calibration import calibrate  # noqa: E402


if __name__ == "__main__":
    print(calibrate())

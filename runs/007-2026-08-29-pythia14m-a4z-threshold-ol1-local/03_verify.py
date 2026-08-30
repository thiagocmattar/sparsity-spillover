"""Re-run Run 007 terminal verification without modifying immutable attempts."""

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from verification import verify_cohort  # noqa: E402


if __name__ == "__main__":
    print(verify_cohort())

"""Execute the approved four-condition cohort after launch confirmation."""

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pipeline import run_cohort  # noqa: E402


if __name__ == "__main__":
    for attempt_dir in run_cohort():
        print(attempt_dir)

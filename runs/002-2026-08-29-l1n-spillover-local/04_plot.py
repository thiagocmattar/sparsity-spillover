"""Regenerate the approved terminal figure from verified artifacts."""

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from plotting import generate_figure  # noqa: E402


if __name__ == "__main__":
    print(generate_figure())

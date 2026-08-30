"""Generate append-only epsilon-1e-2 counterparts to Figures 01-03."""

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from eps1e2_figures import generate_all  # noqa: E402


if __name__ == "__main__":
    for path in generate_all():
        print(path)

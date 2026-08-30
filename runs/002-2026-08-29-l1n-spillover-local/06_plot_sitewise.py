"""Generate the append-only four-panel sitewise near-zero figure."""

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from sitewise_figure import generate_sitewise_figure  # noqa: E402


if __name__ == "__main__":
    print(generate_sitewise_figure())

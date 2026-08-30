"""Generate the append-only attention-output near-zero figure."""

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from attention_output_figure import generate_attention_output_figure  # noqa: E402


if __name__ == "__main__":
    print(generate_attention_output_figure())

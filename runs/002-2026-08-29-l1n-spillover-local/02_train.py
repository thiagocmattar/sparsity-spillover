"""Execute, verify, and plot the pre-approved ten-condition local cohort."""

from datetime import datetime, timezone
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from plotting import generate_figure  # noqa: E402
from run_config import RUN_DIR, write_json  # noqa: E402
from training import run_cohort  # noqa: E402
from verification import verify_cohort  # noqa: E402


if __name__ == "__main__":
    started_at = datetime.now(timezone.utc).isoformat()
    write_json(
        RUN_DIR / "artifacts" / "driver.json",
        {"status": "running", "started_at": started_at, "command": "python 02_train.py"},
    )
    try:
        for attempt_dir in run_cohort():
            print(attempt_dir, flush=True)
        summary_path = verify_cohort()
        print(summary_path, flush=True)
        figure_path = generate_figure(summary_path)
        print(figure_path, flush=True)
        write_json(
            RUN_DIR / "artifacts" / "driver.json",
            {
                "status": "completed",
                "started_at": started_at,
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "command": "python 02_train.py",
                "verification": str(summary_path.relative_to(RUN_DIR)).replace("\\", "/"),
                "figure": str(figure_path.relative_to(RUN_DIR)).replace("\\", "/"),
            },
        )
    except BaseException as error:
        write_json(
            RUN_DIR / "artifacts" / "driver.json",
            {
                "status": "failed",
                "started_at": started_at,
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "command": "python 02_train.py",
                "failure": {"type": type(error).__qualname__, "message": str(error)},
            },
        )
        raise

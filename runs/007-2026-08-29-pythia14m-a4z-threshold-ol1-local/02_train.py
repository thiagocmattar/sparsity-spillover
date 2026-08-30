"""Execute and verify the launch-approved five-condition local cohort."""

from datetime import datetime, timezone
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

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
        verification = verify_cohort()
        print(verification, flush=True)
        write_json(
            RUN_DIR / "artifacts" / "driver.json",
            {
                "status": "completed",
                "started_at": started_at,
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "command": "python 02_train.py",
                "verification": verification.relative_to(RUN_DIR).as_posix(),
            },
        )
    except BaseException as error:
        failure = {"type": type(error).__qualname__, "message": str(error)}
        write_json(
            RUN_DIR / "artifacts" / "driver.json",
            {
                "status": "failed",
                "started_at": started_at,
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "command": "python 02_train.py",
                "failure": failure,
            },
        )
        write_json(RUN_DIR / "artifacts" / "progress.json", {"status": "failed", "failure": failure})
        raise


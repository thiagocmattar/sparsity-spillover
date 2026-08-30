"""Read-only progress summary for the Analysis 006 local evaluator."""

from __future__ import annotations

import json
from pathlib import Path
from statistics import median


ANALYSIS_DIR = Path(__file__).resolve().parent
PROGRESS_PATH = ANALYSIS_DIR / "artifacts" / "progress.jsonl"
RESULT_PATH = ANALYSIS_DIR / "teal_all_variants.json"
TOTAL_NEW_POINTS = 130


def main() -> None:
    if RESULT_PATH.exists():
        result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
        print(
            json.dumps(
                {
                    "status": result.get("status"),
                    "completed_new_points": result.get("source_counts", {}).get("new_points"),
                    "total_new_points": TOTAL_NEW_POINTS,
                    "result": str(RESULT_PATH),
                },
                sort_keys=True,
            )
        )
        return
    if not PROGRESS_PATH.exists():
        print(json.dumps({"status": "not_started", "completed_new_points": 0}))
        return
    rows = [
        json.loads(line)
        for line in PROGRESS_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    protocols = {row["protocol_sha256"] for row in rows}
    if len(protocols) != 1:
        raise ValueError("Progress contains multiple protocol identities.")
    durations = [float(row["evaluation_seconds"]) for row in rows]
    last = rows[-1]
    remaining = TOTAL_NEW_POINTS - len(rows)
    print(
        json.dumps(
            {
                "status": "running_or_interrupted",
                "completed_new_points": len(rows),
                "total_new_points": TOTAL_NEW_POINTS,
                "last_condition_id": last["condition_id"],
                "last_target_sparsity": last["target_sparsity"],
                "last_validation_loss": last["validation"]["loss"],
                "last_R_model": last["logical_products"]["R_model"],
                "last_input_tokens_per_second": last["input_tokens_per_second"],
                "median_point_seconds": median(durations),
                "estimated_remaining_seconds": median(durations) * remaining,
                "protocol_sha256": next(iter(protocols)),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

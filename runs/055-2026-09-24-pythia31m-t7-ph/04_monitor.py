"""Read-only progress and ETC from persistent training events."""
import json
from pathlib import Path
from statistics import median
import time
from run_config import RUN_DIR, load_config


def main():
    config = load_config()
    for attempt in sorted((RUN_DIR/"artifacts/attempts").glob("*")):
        if not (attempt/"manifest.json").is_file():
            continue
        manifest = json.loads((attempt/"manifest.json").read_text())
        events = []
        event_path = attempt/"events.jsonl"
        if event_path.exists():
            for line in event_path.read_text().splitlines():
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    pass  # A concurrently appended final line may be incomplete.
        train = [r for r in events if r.get("event") == "train"]
        if train:
            last = train[-1]
            seconds = median(r["step_wall_seconds"] for r in train[-20:])
            print(json.dumps(dict(condition=manifest["condition"]["id"], status=manifest["status"],
                step=last["step"], target=config["training"]["max_steps"], loss=last["task_loss"],
                tokens_per_second=2097152/seconds,
                remaining_training_seconds=max(0,config["training"]["max_steps"]-last["step"])*seconds,
                stale_seconds=time.time()-event_path.stat().st_mtime)))
        else:
            print(json.dumps(dict(condition=manifest["condition"]["id"],status=manifest["status"],step=0)))


if __name__ == "__main__":
    main()

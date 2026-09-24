"""Sequential one-GPU execution with a fixed deadline and transfer reserve."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from run_config import RUN_DIR, condition_specs, load_config, write_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--deadline", required=True, help="UTC ISO8601 provider-stop deadline")
    p.add_argument("--tag", required=True)
    args = p.parse_args()
    deadline = datetime.fromisoformat(args.deadline.replace("Z","+00:00")).timestamp()
    if not args.tag.replace("-", "").isalnum():
        raise ValueError("Simple unique tag required")
    config = load_config()
    rows = condition_specs(config)
    log_root = RUN_DIR/"artifacts/pipeline"/args.tag
    log_root.mkdir(parents=True, exist_ok=False)
    state = dict(status="running", deadline=args.deadline, completed_conditions=[])
    write_json(log_root/"status.json", state)
    def execute(name, command):
        remaining = deadline-time.time()-3600
        if remaining <= 0:
            raise RuntimeError("One-hour retrieval reserve reached")
        with (log_root/(name+".log")).open("x") as log:
            # A separate group includes any CUDA benchmark child processes.
            child = subprocess.Popen([sys.executable,*command],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            try:
                code = child.wait(timeout=remaining)
                if code:
                    raise subprocess.CalledProcessError(code, child.args)
            finally:
                if child.poll() is None:
                    os.killpg(child.pid,signal.SIGTERM)
                    try:
                        child.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid,signal.SIGKILL)
                        child.wait()
    try:
        predictions = {}
        for row in rows:
            calibration_tag = args.tag+"-"+row["id"]
            command = [str(RUN_DIR/"05_preflight.py"),"--condition",row["id"],"--attempt",calibration_tag]
            if row == rows[-1]:
                command.append("--diagnostics")
            execute("preflight-"+row["id"], command)
            result = json.loads((RUN_DIR/"prelaunch"/("calibration-"+calibration_tag)/"result.json").read_text())
            if result["status"] != "passed":
                raise RuntimeError("Production preflight failed")
            predictions[row["id"]] = result["median_boundary_seconds"]*config["training"]["max_steps"]
        # Thirty percent training contingency, 45 minutes latency, one hour retrieval.
        forecast = 1.3*sum(predictions.values())+2700+3600
        write_json(log_root/"forecast.json",dict(training_seconds=predictions,total_seconds=forecast,
                                                remaining_seconds=deadline-time.time()))
        if forecast > deadline-time.time():
            raise RuntimeError("Measured ETC exceeds the approved deadline before scientific training")
        for row in rows:
            state["current_condition"] = row["id"]
            write_json(log_root/"status.json",state)
            execute("train-"+row["id"],[str(RUN_DIR/"02_train.py"),"--worker",row["id"]])
            progress = json.loads((RUN_DIR/"artifacts/workers"/row["id"]/"progress.json").read_text())
            attempt = RUN_DIR/"artifacts/attempts"/progress["attempts"][0]
            execute("verify-"+row["id"],[str(RUN_DIR/"03_verify.py"),"--attempt",str(attempt)])
            state["completed_conditions"].append(row["id"])
        execute("verify",[str(RUN_DIR/"03_verify.py")])
        execute("latency",[str(RUN_DIR/"09_latency_grid.py"),"--tag",args.tag])
        state.update(status="completed",current_condition=None)
    except BaseException as error:
        state.update(status="failed",error=str(error))
        raise
    finally:
        state["finished_at"] = datetime.now(timezone.utc).isoformat()
        write_json(log_root/"status.json",state)


if __name__ == "__main__":
    main()

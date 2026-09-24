"""Run one approved condition per GPU, with concurrent production preflights."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import os
import signal
import subprocess
import sys
from threading import Lock
import time
from run_config import RUN_DIR, condition_specs, load_config, write_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--workers",nargs="+",required=True)
    p.add_argument("--gpus",nargs="+",type=int,required=True)
    p.add_argument("--tag",required=True)
    p.add_argument("--deadline",required=True)
    args = p.parse_args()
    config = load_config()
    ids = {row["id"] for row in condition_specs(config)}
    if (not set(args.workers) <= ids or len(set(args.workers)) != len(args.workers)
        or len(args.workers) != len(args.gpus) or len(set(args.gpus)) != len(args.gpus)
        or min(args.gpus) < 0 or not args.tag.replace("-","").isalnum()):
        raise ValueError("Distinct approved conditions and one distinct GPU each are required")
    deadline = datetime.fromisoformat(args.deadline.replace("Z","+00:00")).timestamp()
    folder = RUN_DIR/"artifacts/pipeline"/args.tag
    folder.mkdir(parents=True,exist_ok=False)
    lock = Lock()
    state = dict(status="running",deadline=args.deadline,workers={})
    def update(worker,**fields):
        with lock:
            state["workers"].setdefault(worker,{}).update(fields)
            write_json(folder/"status.json",state)
    def execute(worker,gpu,stage,command):
        remaining = deadline-time.time()-3600
        if remaining <= 0:
            raise RuntimeError("One-hour artifact retrieval reserve reached")
        env = dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS="4",MKL_NUM_THREADS="4")
        with (folder/f"{worker}-{stage}.log").open("x") as log:
            child = subprocess.Popen([sys.executable,*map(str,command)],env=env,stdout=log,
                stderr=subprocess.STDOUT,start_new_session=True)
            update(worker,stage=stage,gpu=gpu,pid=child.pid)
            try:
                code = child.wait(timeout=remaining)
                if code:
                    raise subprocess.CalledProcessError(code,child.args)
            finally:
                if child.poll() is None:
                    os.killpg(child.pid,signal.SIGTERM)
                    try:
                        child.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid,signal.SIGKILL)
                        child.wait()
    def preflight(pair):
        worker,gpu = pair
        tag = args.tag+"-"+worker
        execute(worker,gpu,"preflight",[RUN_DIR/"05_preflight.py","--condition",worker,
                "--attempt",tag,"--diagnostics"])
        result = json.loads((RUN_DIR/"prelaunch"/("calibration-"+tag)/"result.json").read_text())
        if result["status"] != "passed":
            raise RuntimeError("Production preflight failed: "+worker)
        prediction = result["median_boundary_seconds"]*config["training"]["max_steps"]
        update(worker,stage="preflight_passed",predicted_training_seconds=prediction)
        return prediction
    def train(pair):
        worker,gpu = pair
        execute(worker,gpu,"training",[RUN_DIR/"02_train.py","--worker",worker])
        progress = json.loads((RUN_DIR/"artifacts/workers"/worker/"progress.json").read_text())
        attempt = RUN_DIR/"artifacts/attempts"/progress["attempts"][0]
        execute(worker,gpu,"verification",[RUN_DIR/"03_verify.py","--attempt",attempt])
        update(worker,stage="verified",attempt=attempt.name)
    try:
        pairs = list(zip(args.workers,args.gpus))
        with ThreadPoolExecutor(max_workers=len(pairs)) as pool:
            predictions = list(pool.map(preflight,pairs))
        forecast = 1.3*max(predictions)+600+3600
        write_json(folder/"forecast.json",dict(training_seconds=dict(zip(args.workers,predictions)),
            total_seconds=forecast,remaining_seconds=deadline-time.time(),parallel=True))
        if forecast > deadline-time.time():
            raise RuntimeError("Parallel training forecast exceeds deadline")
        with ThreadPoolExecutor(max_workers=len(pairs)) as pool:
            list(pool.map(train,pairs))
        state["status"] = "completed"
    except BaseException as error:
        state.update(status="failed",error=str(error))
        raise
    finally:
        state["finished_at"] = datetime.now(timezone.utc).isoformat()
        write_json(folder/"status.json",state)


if __name__ == "__main__":
    main()

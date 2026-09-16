"""Detached launches for the five already authorized Run 032 workers."""
import argparse
import concurrent.futures
import importlib.util
import json
from pathlib import Path
import shlex

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("run032_remote", HERE / "07_remote.py")
remote = importlib.util.module_from_spec(spec)
spec.loader.exec_module(remote)


def launch(pod, phase):
    with remote.connect(pod["id"]) as client:
        prefix = f"cd {remote.REMOTE}; export PYTHONPATH={remote.REMOTE}/{remote.RUN}:{remote.REMOTE}/src; "
        if phase == "preflight":
            command = f"set -e; test -f /workspace/run032-ready; {prefix}{remote.PYTHON} {remote.RUN}/05_remote_preflight.py"
        else:
            with client.open_sftp() as sftp:
                with sftp.open(f"{remote.REMOTE}/{remote.RUN}/prelaunch/remote-preflight.json") as handle:
                    result = json.load(handle)
            receipt = json.loads((HERE / "prelaunch/source-receipt.json").read_text())
            if result["status"] != "passed" or not all(result["checks"].values()):
                raise RuntimeError("Exact preflight has not passed")
            current = remote.command(client, f"cd {remote.REMOTE} && git rev-parse HEAD && git status --porcelain").splitlines()
            if current != [receipt["git_commit"]]:
                raise RuntimeError(f"Source checkout differs: {current}")
            condition = shlex.quote(pod["condition"])
            command = (
                f"set -e; {prefix}"
                f"{remote.PYTHON} {remote.RUN}/02_train.py --worker {condition}; "
                f"{remote.PYTHON} {remote.RUN}/03_verify.py --condition {condition}; "
                f"tar -cf /workspace/run032-results.tar -C {remote.REMOTE}/{remote.RUN} artifacts; "
                "sha256sum /workspace/run032-results.tar > /workspace/run032-results.sha256"
            )
        label = "training" if phase == "training" else "preflight"
        script = f"#!/bin/bash\nset -uo pipefail\n( {command} )\nstatus=$?\necho $status > /workspace/run032-{label}.exit\nexit $status\n"
        remote.command(client, f"test ! -e /workspace/run032-{label}.pid")
        with client.open_sftp() as sftp:
            with sftp.open(f"/workspace/run032-{label}.sh", "w") as handle:
                handle.write(script)
        remote.command(client, f"nohup bash /workspace/run032-{label}.sh > /workspace/run032-{label}.log 2>&1 < /dev/null & echo $! > /workspace/run032-{label}.pid")
        print(json.dumps({"pod": pod["id"], "condition": pod["condition"], "phase": phase, "status": "launched"}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("preflight", "training"))
    parser.add_argument("--pod", action="append", required=True)
    args = parser.parse_args()
    pods = json.loads((HERE / "prelaunch/pods.json").read_text(encoding="utf-8-sig"))
    selected = [pod for pod in pods if pod["id"] in args.pod]
    if len(selected) != len(set(args.pod)):
        raise ValueError("Unknown worker")
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
        list(pool.map(lambda pod: launch(pod, args.phase), selected))


if __name__ == "__main__":
    main()

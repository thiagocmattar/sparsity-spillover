"""Distribute Run 032's one verified cache seed through encrypted RunPod transfer."""
import concurrent.futures
import importlib.util
import json
from pathlib import Path
import re
import shlex
import time

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("run032_remote", HERE / "07_remote.py")
remote = importlib.util.module_from_spec(spec)
spec.loader.exec_module(remote)


def start_copy(source, target):
    tag = target["id"]
    with remote.connect(source["id"]) as sender:
        remote.command(sender, "test -f /workspace/run032-cache-ready")
        path = remote.REMOTE + "/data/tokenized/minipile-pythia-14m-full"
        text = f"/workspace/run032-runpodctl send {shlex.quote(path)} > /workspace/run032-send-{tag}.log 2>&1; echo $? > /workspace/run032-send-{tag}.exit"
        remote.command(sender, f"nohup bash -c {shlex.quote(text)} > /dev/null 2>&1 < /dev/null &")
        code = None
        for _ in range(20):
            line = remote.command(sender, f"head -n 1 /workspace/run032-send-{tag}.log").strip()
            # Current runpodctl emits the fresh code as its first stdout line.
            if re.fullmatch(r"[A-Za-z0-9-]+", line):
                code = line
                break
            time.sleep(1)
        if code is None:
            raise RuntimeError("Sender did not publish a usable transfer code")
    with remote.connect(target["id"]) as receiver:
        text = f"cd /opt/run032-cache-parent && /workspace/run032-runpodctl receive {shlex.quote(code)} > /workspace/run032-receive.log 2>&1; status=$?; echo $status > /workspace/run032-receive.exit; if [ $status -eq 0 ]; then touch /workspace/run032-cache-ready; fi; exit $status"
        remote.command(receiver, f"nohup bash -c {shlex.quote(text)} > /dev/null 2>&1 < /dev/null &")
    return {"source": source["id"], "receiver": target["id"], "status": "transfer_started"}


def main():
    pods = json.loads((HERE / "prelaunch/pods.json").read_text(encoding="utf-8-sig"))
    source = next(p for p in pods if p["condition"] == "a7-h-ol1-kappa-0p5")
    targets = [p for p in pods if p is not source]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda p: start_copy(source, p), targets))
    (HERE / "prelaunch/cache-distribution.json").write_text(json.dumps(results, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(results))


if __name__ == "__main__":
    main()

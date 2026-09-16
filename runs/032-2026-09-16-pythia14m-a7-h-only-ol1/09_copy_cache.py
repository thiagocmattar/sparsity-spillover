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
    parent = "/opt/run032-cache-parent/minipile-pythia-14m-full"
    with remote.connect(target["id"]) as receiver:
        if remote.command(receiver, "test -f /workspace/run032-cache-ready && echo ready || true").strip():
            return {"source": source["id"], "receiver": tag, "status": "already_transferred"}
    with remote.connect(source["id"]) as sender:
        remote.command(sender, "test -f /workspace/run032-cache-ready")
        with sender.open_sftp() as sftp:
            small = {}
            for name in ["train/metadata.json", "validation/metadata.json", "validation/tokens.int32.bin"]:
                with sftp.open(parent + "/" + name, "rb") as handle:
                    handle.prefetch()
                    small[name] = handle.read()
        with remote.connect(target["id"]) as receiver:
            remote.command(receiver, f"mkdir -p {parent}/train {parent}/validation; test ! -f {parent}/train/tokens.int32.bin")
            with receiver.open_sftp() as sftp:
                for name, payload in small.items():
                    with sftp.open(parent + "/" + name, "wb") as handle:
                        handle.set_pipelined(True)
                        handle.write(payload)
        path = parent + "/train/tokens.int32.bin"
        text = f"/workspace/run032-runpodctl send {shlex.quote(path)} > /workspace/run032-send-{tag}-direct.log 2>&1; echo $? > /workspace/run032-send-{tag}-direct.exit"
        remote.command(sender, f"nohup bash -c {shlex.quote(text)} > /dev/null 2>&1 < /dev/null &")
        code = None
        for _ in range(20):
            line = remote.command(sender, f"head -n 1 /workspace/run032-send-{tag}-direct.log").strip()
            # Current runpodctl emits the fresh code as its first stdout line.
            if re.fullmatch(r"[A-Za-z0-9-]+", line):
                code = line
                break
            time.sleep(1)
        if code is None:
            raise RuntimeError("Sender did not publish a usable transfer code")
    with remote.connect(target["id"]) as receiver:
        text = f"cd {parent}/train && /workspace/run032-runpodctl receive {shlex.quote(code)} > /workspace/run032-receive-direct.log 2>&1; status=$?; echo $status > /workspace/run032-receive-direct.exit; if [ $status -eq 0 ]; then touch /workspace/run032-cache-ready; fi; exit $status"
        remote.command(receiver, f"nohup bash -c {shlex.quote(text)} > /dev/null 2>&1 < /dev/null &")
    return {"source": source["id"], "receiver": target["id"], "status": "transfer_started"}


def main():
    pods = json.loads((HERE / "prelaunch/pods.json").read_text(encoding="utf-8-sig"))
    source = next(p for p in pods if p["condition"] == "a7-h-ol1-kappa-0p5")
    targets = [p for p in pods if p is not source]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda p: start_copy(source, p), targets))
    (HERE / "prelaunch/cache-distribution-direct.json").write_text(json.dumps(results, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(results))


if __name__ == "__main__":
    main()

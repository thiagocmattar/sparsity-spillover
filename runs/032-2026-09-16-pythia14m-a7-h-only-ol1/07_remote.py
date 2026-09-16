"""Run 032 SSH transport; no provisioning, scheduling, or scientific choices."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import time

import paramiko

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CLI = ROOT / "tmp/runpodctl-v2.12.0.exe"
KEY = Path.home() / ".runpod/ssh/runpodctl-ssh-key"
REMOTE = "/workspace/sparsity-spillover"
RUN = "runs/" + HERE.name
PYTHON = "/workspace/run032-venv/bin/python"


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def connect(pod):
    value = json.loads(subprocess.check_output([str(CLI), "pod", "get", pod], text=True))
    if not value.get("name", "").startswith("run032-"):
        raise ValueError("Not a Run 032 Pod")
    host = value.get("publicIp")
    port = value.get("portMappings", {}).get("22")
    if not host or not port:
        ports = (value.get("runtime") or {}).get("ports", [])
        match = next((p for p in ports if p.get("privatePort") == 22 and p.get("isIpPublic")), None)
        if match:
            host, port = match["ip"], match["publicPort"]
    if not host or not port:
        raise RuntimeError("Public SSH not ready")
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(host, port=int(port), username="root", key_filename=str(KEY),
                   timeout=20, banner_timeout=30, auth_timeout=30)
    client.get_transport().set_keepalive(30)
    return client


def command(client, text):
    _, stdout, stderr = client.exec_command(text)
    out = stdout.read().decode("utf-8", errors="replace")
    err = stderr.read().decode("utf-8", errors="replace")
    status = stdout.channel.recv_exit_status()
    if status:
        raise RuntimeError(f"Remote command exited {status}: {err[-4000:]} {out[-4000:]}")
    return out


def upload(client, local, remote):
    command(client, "mkdir -p " + shlex.quote(str(Path(remote).parent).replace("\\", "/")))
    started = time.monotonic()
    with client.open_sftp() as sftp:
        sftp.put(str(local), remote)
    size = Path(local).stat().st_size
    print(json.dumps({"phase": "uploaded", "file": str(local), "bytes": size,
                      "seconds": round(time.monotonic()-started, 3)}), flush=True)


def prepare(client, pod, preflight):
    bundle = HERE / "prelaunch/source.bundle"
    receipt = json.loads((HERE / "prelaunch/source-receipt.json").read_text())
    upload(client, bundle, "/workspace/run032-source.bundle")
    remote_hash = command(client, "sha256sum /workspace/run032-source.bundle").split()[0]
    if remote_hash != receipt["sha256"]:
        raise RuntimeError("Source bundle transfer mismatch")
    command(client, f"test ! -e {REMOTE} && git clone /workspace/run032-source.bundle {REMOTE} && cd {REMOTE} && git checkout {receipt['git_commit']}")
    if command(client, f"cd {REMOTE} && git rev-parse HEAD").strip() != receipt["git_commit"]:
        raise RuntimeError("Remote commit mismatch")
    if command(client, f"cd {REMOTE} && git status --porcelain").strip():
        raise RuntimeError("Remote source checkout is dirty")
    command(client, f"printf '%s\\n' '{RUN}/prelaunch/' '{RUN}/artifacts/' >> {REMOTE}/.git/info/exclude")
    setup = f"""#!/bin/bash
set -euo pipefail
cd {REMOTE}
bash {RUN}/00_setup_remote.sh
touch /workspace/run032-environment-ready
"""
    with client.open_sftp() as sftp:
        with sftp.open("/workspace/run032-setup.sh", "w") as handle:
            handle.write(setup)
    command(client, "nohup bash /workspace/run032-setup.sh > /workspace/run032-setup.log 2>&1 < /dev/null & echo $! > /workspace/run032-setup.pid")
    for split in ("validation", "train"):
        folder = ROOT / "data/tokenized/minipile-pythia-14m-full" / split
        for name in ("metadata.json", "tokens.int32.bin"):
            local = folder / name
            upload(client, local, REMOTE + "/" + local.relative_to(ROOT).as_posix())
    print(json.dumps({"phase": "source_and_caches_uploaded", "pod": pod}), flush=True)
    readiness = f"""#!/bin/bash
set -euo pipefail
while [ ! -f /workspace/run032-environment-ready ]; do
  if ! kill -0 $(cat /workspace/run032-setup.pid) 2>/dev/null; then echo SETUP_FAILED; exit 2; fi
  sleep 10
done
cd {REMOTE}
export PYTHONPATH={REMOTE}/{RUN}:{REMOTE}/src
{PYTHON} - <<'PY'
import json,sys,torch,numpy as np,transformers
from run_config import load_config,load_verified_caches,build_schedule,EXPECTED_SCHEDULE_SHA256,approved_identity
c=load_config()
realized={{'python':f'{{sys.version_info.major}}.{{sys.version_info.minor}}','torch':torch.__version__.split('+')[0],'transformers':transformers.__version__,'cuda_runtime':torch.version.cuda}}
assert realized==c['runtime'],realized
train,val,tm,vm,seconds=load_verified_caches(c,np=np)
assert build_schedule(c,tm,np=np)[1]==EXPECTED_SCHEDULE_SHA256
print(json.dumps({{'runtime':realized,'identity':approved_identity(),'cache_verification_seconds':seconds,'gpu':torch.cuda.get_device_name(0)}}))
PY
{"" if not preflight else PYTHON + " " + RUN + "/05_remote_preflight.py"}
touch /workspace/run032-ready
"""
    with client.open_sftp() as sftp:
        with sftp.open("/workspace/run032-readiness.sh", "w") as handle:
            handle.write(readiness)
    command(client, "nohup bash /workspace/run032-readiness.sh > /workspace/run032-readiness.log 2>&1 < /dev/null & echo $! > /workspace/run032-readiness.pid")


def status(client):
    text = """python3 - <<'PY'
import json,pathlib,subprocess
p=pathlib.Path('/workspace')
result={'ready':(p/'run032-ready').exists(),'environment_ready':(p/'run032-environment-ready').exists()}
for phase in ['setup','readiness','training']:
 f=p/f'run032-{phase}.log'
 if f.exists():result[phase+'_tail']=f.read_text(errors='replace').splitlines()[-3:]
 pid=p/f'run032-{phase}.pid'
 if pid.exists():result[phase+'_pid']=pid.read_text().strip()
result['gpu']=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.used,memory.total,utilization.gpu','--format=csv,noheader'],text=True).strip()
result['disk']=subprocess.check_output(['df','-h','/workspace'],text=True).strip()
print(json.dumps(result))
PY"""
    print(command(client, text), flush=True)
    print(command(client, f"if test -f /workspace/run032-environment-ready; then cd {REMOTE}; {PYTHON} {RUN}/04_monitor.py; fi"), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "status", "command", "get"))
    parser.add_argument("--pod", required=True)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--command-file", type=Path)
    parser.add_argument("--remote")
    parser.add_argument("--local", type=Path)
    args = parser.parse_args()
    with connect(args.pod) as client:
        if args.action == "prepare":
            prepare(client, args.pod, args.preflight)
        elif args.action == "status":
            status(client)
        elif args.action == "command":
            print(command(client, args.command_file.read_text(encoding="utf-8")), flush=True)
        elif args.action == "get":
            args.local.parent.mkdir(parents=True, exist_ok=True)
            with client.open_sftp() as sftp:
                sftp.get(args.remote, str(args.local))
            print(json.dumps({"path": str(args.local), "bytes": args.local.stat().st_size, "sha256": sha(args.local)}))


if __name__ == "__main__":
    main()

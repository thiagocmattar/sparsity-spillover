"""Run063 transport retry: confirmed SSH and streamed local-disk ingress.

Scientific bundle and executable scripts remain byte-identical. Uses only the
owned lease; leaves existing remote/workstation deadline guards in place.
"""
import argparse
import importlib
import json
from pathlib import Path
import subprocess
import sys
import time
import paramiko

RUN = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RUN))
original = importlib.import_module("06_transfer")


def connect():
    lease = original.read(RUN / "prelaunch/lease-001.json")
    pod = json.loads(subprocess.check_output(
        [str(original.CLI), "pod", "get", lease["pod"]["id"]], text=True))
    if pod["id"] != lease["pod"]["id"] or pod["name"] != lease["pod"]["name"]:
        raise ValueError("Owned Pod identity mismatch")
    info = pod["ssh"]
    if not info.get("ip"):
        confirmed = original.read(RUN / "prelaunch/ssh-confirmed-001.json")
        if confirmed["pod_id"] != pod["id"] or confirmed["pod_name"] != pod["name"]:
            raise ValueError("Confirmed route belongs to another Pod")
        info = {"ip": confirmed["host"], "port": confirmed["port"],
            "ssh_key": {"path": str(Path.home()/".runpod/ssh/runpodctl-ssh-key")}}
    client = paramiko.SSHClient()
    client.load_host_keys(str(RUN / "prelaunch/known_hosts"))
    client.set_missing_host_key_policy(paramiko.RejectPolicy())
    client.connect(info["ip"], port=info["port"], username="root",
        key_filename=[info["ssh_key"]["path"], str(Path.home()/".ssh/id_ed25519_runpod")],
        timeout=20, auth_timeout=20, banner_timeout=20,
        look_for_keys=False, allow_agent=False)
    client.get_transport().set_keepalive(20)
    return client, lease


def prepare(client, lease):
    receipt = original.read(RUN / "bundles/input-002.receipt.json")
    remote = original.REMOTE
    control = original.CONTROL
    original.execute(client, f"test ! -e {remote} && test ! -e {control}/pipeline.pid && test -s {control}/guard.log")
    started = time.monotonic()
    stdin, stdout, stderr = client.exec_command("cat > /opt/run063-input.tar", timeout=60)
    sent = 0
    next_report = 10 * 1024**2
    with (RUN / receipt["path"]).open("rb") as source:
        while chunk := source.read(1024**2):
            stdin.channel.sendall(chunk)
            sent += len(chunk)
            if sent >= next_report:
                elapsed = time.monotonic()-started
                print(json.dumps({"stage":"upload","bytes":sent,"total":receipt["bytes"],
                    "seconds":elapsed,"remaining_seconds":(receipt["bytes"]-sent)/(sent/elapsed)}), flush=True)
                next_report += 10 * 1024**2
    stdin.channel.shutdown_write()
    stdout.read()
    error = stderr.read().decode(errors="replace")
    if stdout.channel.recv_exit_status():
        raise RuntimeError(error)
    upload_seconds = time.monotonic()-started
    if original.execute(client, "sha256sum /opt/run063-input.tar").split()[0] != receipt["sha256"]:
        raise RuntimeError("Fast ingress archive hash mismatch")
    original.execute(client, "cp /opt/run063-input.tar /workspace/run063-input.tar", timeout=300)
    if original.execute(client, "sha256sum /workspace/run063-input.tar").split()[0] != receipt["sha256"]:
        raise RuntimeError("Persistent archive hash mismatch")
    original.execute(client, f"mkdir {remote} && tar -xf /opt/run063-input.tar -C {remote}", timeout=300)
    with client.open_sftp() as sftp:
        with sftp.open(remote+"/provenance/deployment.json", "w") as handle:
            handle.write(json.dumps({"source_checkout":lease["source_checkout"],"input_bundle":receipt,
                "pod_id":lease["pod"]["id"],"pod_name":lease["pod"]["name"],
                "transport_retry":original.record(__file__)}))
    wrapper = (f"cd {remote}; bash {remote}/04_setup.sh || exit $?; "
        f"bash {remote}/05_execute.sh smoke {lease['deadline_epoch']} || exit $?; "
        f"bash {remote}/05_execute.sh scientific {lease['deadline_epoch']}")
    wrapper = f"({wrapper}); code=$?; printf '%s\\n' \"$code\" > {control}/pipeline.exit"
    original.execute(client, f"nohup setsid bash -c {original.shlex.quote(wrapper)} > {control}/pipeline.log 2>&1 < /dev/null & echo $! > {control}/pipeline.pid")
    original.write(RUN/"prelaunch/upload-002.json", {"source":receipt,"upload_seconds":upload_seconds,
        "archive_sha256_verified":True,"persistent_archive_sha256_verified":True,
        "detached":True,"retry_transport":original.record(__file__)})
    print(json.dumps({"stage":"pipeline_started","upload_seconds":upload_seconds}), flush=True)


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("action", choices=["prepare","status","retrieve"])
    args=parser.parse_args()
    client,lease=connect()
    try:
        if args.action=="prepare": prepare(client,lease)
        elif args.action=="status": original.status(client)
        else: original.retrieve(client)
    finally: client.close()

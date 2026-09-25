"""Run056 attempt-001 SSH transport; infrastructure only."""
import json
from pathlib import Path
import sys
import paramiko

RUN = Path(__file__).resolve().parents[1]

def connect():
    pod = json.loads((RUN / 'prelaunch/pod-ready-001.json').read_text())
    direct = pod['ssh']['direct']
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(direct['host'], port=direct['port'], username=direct.get('username', 'root'),
                   key_filename=str(Path.home() / '.runpod/ssh/runpodctl-ssh-key'),
                   timeout=20, banner_timeout=30, auth_timeout=20)
    transport = client.get_transport()
    transport.set_keepalive(30)
    fingerprint = transport.get_remote_server_key().get_fingerprint().hex()
    record = RUN / 'prelaunch/ssh-host-fingerprint-001.json'
    if record.exists():
        assert json.loads(record.read_text())['fingerprint'] == fingerprint
    else:
        record.write_text(json.dumps({'fingerprint': fingerprint, 'host': direct['host']}))
    return client

def main():
    client = connect()
    try:
        if sys.argv[1] == 'exec':
            command = sys.stdin.buffer.read().decode('utf-8-sig').replace('\r\n', '\n')
            _, stdout, stderr = client.exec_command(command, timeout=300)
            sys.stdout.buffer.write(stdout.read())
            sys.stderr.buffer.write(stderr.read())
            raise SystemExit(stdout.channel.recv_exit_status())
        with client.open_sftp() as sftp:
            if sys.argv[1] == 'put':
                sftp.put(sys.argv[2], sys.argv[3])
            elif sys.argv[1] == 'get':
                sftp.get(sys.argv[2], sys.argv[3])
            else:
                raise ValueError('Unknown transport action')
    finally:
        client.close()

if __name__ == '__main__':
    main()

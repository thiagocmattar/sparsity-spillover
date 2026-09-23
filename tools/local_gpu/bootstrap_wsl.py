"""Install the dedicated distro and complete its GPU setup (run while idle)."""
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import tarfile
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DISTRO = 'SparsityGPU'
TASK_ROOT = Path(os.environ['LOCALAPPDATA']) / 'sparsity-spillover'
WSL_CONF = '''[automount]
enabled=false
mountFsTab=false
[interop]
enabled=false
appendWindowsPath=false
[user]
default=researcher
[boot]
systemd=false
'''


def command(args, data=None):
    result = subprocess.run(args, input=data, capture_output=True)
    def decode(value):
        return value.decode('utf-16-le' if b'\0' in value[:200] else 'utf-8', errors='replace').replace('\r', '').lstrip('\ufeff')
    output = decode(result.stdout)
    if result.returncode:
        raise RuntimeError(f'{args}: exit {result.returncode}\n{output}\n{decode(result.stderr)}')
    return output


def linux(script, data=None):
    return command(['wsl.exe', '-d', DISTRO, '-u', 'root', '--exec', 'bash', '-c', script], data)


def main():
    receipt = json.loads((HERE / 'host-setup.json').read_text())
    image = TASK_ROOT / 'gpu-setup' / receipt['ubuntu_image']['filename']
    with image.open('rb') as source:
        assert hashlib.file_digest(source, 'sha256').hexdigest() == receipt['ubuntu_image']['sha256']
    listing = command(['wsl.exe', '--list', '--quiet'])
    if DISTRO not in listing.splitlines():
        location = TASK_ROOT / 'wsl'
        if location.exists() and any(location.iterdir()):
            raise RuntimeError(f'Refusing to replace nonempty distro location: {location}')
        print(command(['wsl.exe', '--import', DISTRO, str(location), str(image), '--version', '2']), flush=True)
    print(linux('set -e; source /etc/os-release; test "$ID:$VERSION_ID" = ubuntu:24.04; '
                'id researcher >/dev/null 2>&1 || useradd --create-home --shell /bin/bash researcher; '
                'if test -e /etc/wsl.conf && ! test -e /etc/wsl.conf.before-local-gpu; then cp /etc/wsl.conf /etc/wsl.conf.before-local-gpu; fi; '
                'cat > /etc/wsl.conf', WSL_CONF.encode()), flush=True)
    command(['wsl.exe', '--terminate', DISTRO])
    time.sleep(8)
    isolation = linux('set -e; '
                      'test ! -e /mnt/c/Windows; '
                      'if findmnt -rn -o TARGET | grep -Eq "^/mnt/[a-z]($|/)"; then exit 1; fi; '
                      'test -z "${WSL_INTEROP:-}"; '
                      'cat /etc/wsl.conf; id researcher; /usr/lib/wsl/lib/nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version --format=csv,noheader')
    print(isolation, flush=True)
    linux('install -d /opt/sparsity-gpu/setup /opt/sparsity-gpu/logs')
    payload = io.BytesIO()
    sources = {'setup.sh': HERE / 'setup.sh', 'smoke.py': HERE / 'smoke.py',
               'pip-freeze.txt': ROOT / 'runs/049-2026-09-22-pythia70m-short-row-limits/provenance/pip-freeze.txt'}
    with tarfile.open(fileobj=payload, mode='w:gz') as archive:
        for name, path in sources.items():
            # Text transfer only; keep shell files LF even on a CRLF checkout.
            content = path.read_text(encoding='utf-8').encode('utf-8')
            member = tarfile.TarInfo(name)
            member.size = len(content)
            member.mode = 0o644
            archive.addfile(member, io.BytesIO(content))
    linux('tar -xzf - -C /opt/sparsity-gpu/setup', payload.getvalue())
    report = {'status': 'setup running', 'distribution': DISTRO,
              'started_utc': datetime.now(timezone.utc).isoformat(),
              'windows_drive_automount': False, 'windows_executable_interop': False,
              'default_user': 'researcher', 'isolation_checks': isolation,
              'setup_source_sha256': {name: hashlib.sha256(path.read_text(encoding='utf-8').encode()).hexdigest()
                                      for name, path in sources.items()}}
    (HERE / 'wsl-bootstrap.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    # Keep the Windows WSL client alive. Linux nohup alone did not keep this
    # distro running after the short-lived Windows caller exited.
    result = subprocess.run(['wsl.exe', '-d', DISTRO, '-u', 'root', '--exec', 'bash', '-c',
                             'bash /opt/sparsity-gpu/setup/setup.sh; result=$?; '
                             'echo $result > /opt/sparsity-gpu/logs/setup.exit; exit $result'])
    report.update(status='ready' if result.returncode == 0 else 'setup failed',
                  exit_code=result.returncode,
                  finished_utc=datetime.now(timezone.utc).isoformat())
    (HERE / 'wsl-bootstrap.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report), flush=True)
    result.check_returncode()


if __name__ == '__main__':
    main()

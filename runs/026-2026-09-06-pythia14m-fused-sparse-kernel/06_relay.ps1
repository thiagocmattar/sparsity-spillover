$ErrorActionPreference='Stop'
$run026Root=(Resolve-Path "$PSScriptRoot/../..").Path
$run026Cli=Join-Path $run026Root 'tmp/runpodctl-v2.12.0.exe'
$run026Archive=Join-Path $PSScriptRoot 'bundles/initial-002/payload.tar.gz'
$run026Scratch=Join-Path $run026Root 'tmp/run026-relay-001'
$null=New-Item -ItemType Directory -Path $run026Scratch
$run026Code='2601-'+[Guid]::NewGuid().ToString('N')
$run026Send=Start-Process $run026Cli -WindowStyle Hidden -ArgumentList @('send',"`"$run026Archive`"",'--code',$run026Code) -PassThru -RedirectStandardOutput "$run026Scratch/send.out" -RedirectStandardError "$run026Scratch/send.err"
$run026Remote="mkdir -p /workspace/run026-relay-001 && cd /workspace/run026-relay-001 && runpodctl receive $run026Code"
$run026Hosts=Join-Path $PSScriptRoot 'launch-control/rtx5090-001/known_hosts'
$run026Receive=Start-Process ssh.exe -WindowStyle Hidden -ArgumentList @('-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o',"UserKnownHostsFile=$run026Hosts",'-i','C:/Users/thima/.runpod/ssh/runpodctl-ssh-key','-p','31458','root@174.94.157.109',"`"$run026Remote`"") -PassThru -RedirectStandardOutput "$run026Scratch/receive.out" -RedirectStandardError "$run026Scratch/receive.err"
[ordered]@{started_utc=[DateTimeOffset]::UtcNow.ToString('o');send_pid=$run026Send.Id;receive_pid=$run026Receive.Id;payload_bytes=53371675;payload_sha256='1694ecaa4860447740da5f7f0eb6498c6cfe0d8eac795da2dd45f048424b4ae0';pod_id='gc38kxs85gwxra';route='RunPod encrypted one-time relay';raw_logs='untracked tmp/run026-relay-001'} | ConvertTo-Json | Tee-Object -FilePath "$PSScriptRoot/launch-control/rtx5090-001/relay.json"

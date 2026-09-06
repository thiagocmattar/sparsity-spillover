$ErrorActionPreference='Stop'
$run026Root=(Resolve-Path "$PSScriptRoot/../..").Path
$run026Scratch=Join-Path $run026Root 'tmp/run026-relay-001'
$run026Code=(Get-Content "$run026Scratch/send.out" | Where-Object {$_ -match '^2601-[a-f0-9]+-[0-9]+$'} | Select-Object -First 1).Trim()
if (-not $run026Code) { throw 'Actual relay code not yet emitted' }
$run026Remote="cd /workspace/run026-relay-001 && runpodctl receive $run026Code"
$run026Hosts=Join-Path $PSScriptRoot 'launch-control/rtx5090-001/known_hosts'
$run026Receive=Start-Process ssh.exe -WindowStyle Hidden -ArgumentList @('-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o',"UserKnownHostsFile=$run026Hosts",'-i','C:/Users/thima/.runpod/ssh/runpodctl-ssh-key','-p','31458','root@174.94.157.109',"`"$run026Remote`"") -PassThru -RedirectStandardOutput "$run026Scratch/receive2.out" -RedirectStandardError "$run026Scratch/receive2.err"
"receive_pid=$($run026Receive.Id)"

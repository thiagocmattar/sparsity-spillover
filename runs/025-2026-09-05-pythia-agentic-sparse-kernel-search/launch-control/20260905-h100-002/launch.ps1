$ErrorActionPreference = 'Stop'
$run025Root = (Resolve-Path "$PSScriptRoot/../../../..").Path
$run025Run = (Resolve-Path "$PSScriptRoot/../..").Path
$run025Cli = Join-Path $run025Root 'tmp/runpodctl-v2.12.0.exe'
$run025Name = 'run025-calibration-h100-002'
$run025Before = [DateTimeOffset]::UtcNow
$run025Deadline = $run025Before.AddMinutes(46)
$run025Raw = & $run025Cli pod create --name $run025Name --image 'runpod/pytorch@sha256:0a360022e8de4375af99430f84e8b38951acc397252163a37ceac7204d01be35' --cloud-type SECURE --gpu-id 'NVIDIA H100 80GB HBM3' --gpu-count 1 --min-cuda-version 12.8 --container-disk-in-gb 40 --volume-in-gb 80 --volume-mount-path /workspace --ports '22/tcp' --ssh --wait --wait-timeout 60s
if ($LASTEXITCODE -ne 0) { throw 'Discover Pods by exact name before retry: creation may have allocated a Pod.' }
$run025Pod = $run025Raw | ConvertFrom-Json
if (-not $run025Pod.id -or $run025Pod.name -ne $run025Name) { throw 'Unexpected creation response; reconcile immediately' }
$run025Guard = Join-Path $run025Run '06_billing_guard.ps1'
$run025Log = Join-Path $PSScriptRoot 'billing-guard.log'
$run025Args = @('-NoProfile','-ExecutionPolicy','Bypass','-File',"`"$run025Guard`"",'-PodId',$run025Pod.id,'-ExpectedName',$run025Name,'-DeadlineUtc',$run025Deadline.ToString('o'),'-RunpodctlPath',"`"$run025Cli`"",'-LogPath',"`"$run025Log`"")
$run025Process = Start-Process powershell.exe -WindowStyle Hidden -ArgumentList $run025Args -PassThru -RedirectStandardError (Join-Path $PSScriptRoot 'guard-error.log')
$run025Lease = [ordered]@{pod_id=$run025Pod.id; name=$run025Name; requested_utc=$run025Before.ToString('o'); stop_deadline_utc=$run025Deadline.ToString('o'); quoted_gpu_usd_per_hour=3.49; lease_minutes_max=46; pilot_total_usd_max=5; guard_pid=$run025Process.Id; source_commit='afad780'; returned_cost_per_hour=$run025Pod.costPerHr}
$run025Lease | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $PSScriptRoot 'lease.json') -Encoding UTF8
$run025Lease | ConvertTo-Json

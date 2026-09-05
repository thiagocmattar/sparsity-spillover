$ErrorActionPreference = 'Stop'
$run025Root = (Resolve-Path "$PSScriptRoot/../../../..").Path
$run025Run = (Resolve-Path "$PSScriptRoot/../..").Path
$run025Cli = Join-Path $run025Root 'tmp/runpodctl-v2.12.0.exe'
$run025Name = 'run025-calibration-rtx5090-001'
$run025Before = [DateTimeOffset]::UtcNow
$run025Deadline = $run025Before.AddHours(2)
$run025Raw = & $run025Cli pod create --name $run025Name --image 'runpod/pytorch@sha256:0a360022e8de4375af99430f84e8b38951acc397252163a37ceac7204d01be35' --cloud-type COMMUNITY --gpu-id 'NVIDIA GeForce RTX 5090' --gpu-count 1 --min-cuda-version 12.8 --container-disk-in-gb 40 --volume-in-gb 80 --volume-mount-path /workspace --ports '22/tcp' --public-ip --ssh --wait --wait-timeout 60s
if ($LASTEXITCODE -ne 0) {
    throw 'Creation did not return ready. Discover Pods by exact run025 name before any retry; an allocated Pod may still exist.'
}
$run025Pod = $run025Raw | ConvertFrom-Json
if (-not $run025Pod.id -or $run025Pod.name -ne $run025Name) { throw 'Unexpected creation response; reconcile resources immediately' }
$run025Guard = Join-Path $run025Run '06_billing_guard.ps1'
$run025Log = Join-Path $PSScriptRoot 'billing-guard.log'
$run025Args = @('-NoProfile','-ExecutionPolicy','Bypass','-File',"`"$run025Guard`"",'-PodId',$run025Pod.id,'-ExpectedName',$run025Name,'-DeadlineUtc',$run025Deadline.ToString('o'),'-RunpodctlPath',"`"$run025Cli`"",'-LogPath',"`"$run025Log`"")
$run025Process = Start-Process powershell.exe -WindowStyle Hidden -ArgumentList $run025Args -PassThru -RedirectStandardError (Join-Path $PSScriptRoot 'guard-error.log')
$run025Lease = [ordered]@{ pod_id=$run025Pod.id; name=$run025Name; requested_utc=$run025Before.ToString('o'); stop_deadline_utc=$run025Deadline.ToString('o'); quoted_gpu_usd_per_hour=0.69; lease_hours_max=2; pilot_total_usd_max=5; guard_pid=$run025Process.Id; image='runpod/pytorch@sha256:0a360022e8de4375af99430f84e8b38951acc397252163a37ceac7204d01be35'; source_commit='afad780'; returned_cost_per_hour=$run025Pod.costPerHr }
$run025Lease | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $PSScriptRoot 'lease.json') -Encoding UTF8
$run025Lease | ConvertTo-Json

param([string]$Attempt='rtx5090-001',[ValidateSet('COMMUNITY','SECURE')][string]$Cloud='COMMUNITY',[double]$QuotedRate=0.69,[double]$Hours=16)
$ErrorActionPreference='Stop'
if ($Attempt -notmatch '^[a-z0-9-]+$' -or $Hours -le 0 -or $Hours -gt 16 -or $QuotedRate -gt 0.99) { throw 'Outside approved launch envelope' }
$run029Root=(Resolve-Path "$PSScriptRoot/../..").Path
$run029Cli=Join-Path $run029Root 'tmp/runpodctl-v2.12.0.exe'
$run029Folder=Join-Path $PSScriptRoot "launch-control/$Attempt"
$run029Name="run029-$Attempt"
if (Test-Path -LiteralPath $run029Folder) { throw 'Attempt exists; inspect before retry' }
$null=New-Item -ItemType Directory -Path $run029Folder
$run029Start=[DateTimeOffset]::UtcNow
$run029Deadline=$run029Start.AddHours($Hours)
# CLI is needed for Community public-IP and minimum-CUDA constraints not exposed by MCP.
$run029Raw=& $run029Cli pod create --name $run029Name --image 'runpod/pytorch@sha256:0a360022e8de4375af99430f84e8b38951acc397252163a37ceac7204d01be35' --cloud-type $Cloud --gpu-id 'NVIDIA GeForce RTX 5090' --gpu-count 1 --min-cuda-version 12.8 --container-disk-in-gb 20 --volume-in-gb 50 --volume-mount-path /workspace --ports '22/tcp' --public-ip --ssh
if ($LASTEXITCODE -ne 0) { throw 'Create failed: inspect resources by exact name before retry' }
$run029Pod=$run029Raw | ConvertFrom-Json
if (-not $run029Pod.id -or $run029Pod.name -ne $run029Name) { throw 'Unexpected response: inspect control plane' }
$run029Args=@('-NoProfile','-ExecutionPolicy','Bypass','-File',"`"$PSScriptRoot/06_stop_guard.ps1`"",'-PodId',$run029Pod.id,'-ExpectedName',$run029Name,'-DeadlineUtc',$run029Deadline.ToString('o'),'-RunpodctlPath',"`"$run029Cli`"",'-LogPath',"`"$run029Folder/guard.log`"")
$run029Guard=Start-Process powershell.exe -WindowStyle Hidden -ArgumentList $run029Args -PassThru -RedirectStandardError "$run029Folder/guard.err"
$run029Lease=[ordered]@{pod_id=$run029Pod.id;name=$run029Name;start_utc=$run029Start.ToString('o');stop_deadline_utc=$run029Deadline.ToString('o');quoted_rate=$QuotedRate;returned_rate=$run029Pod.costPerHr;max_hours=$Hours;guard_pid=$run029Guard.Id;total_run_budget_usd=20;workspace='/workspace/run029';image='runpod/pytorch@sha256:0a360022e8de4375af99430f84e8b38951acc397252163a37ceac7204d01be35'}
$run029Lease | ConvertTo-Json | Set-Content -LiteralPath "$run029Folder/lease.json" -Encoding UTF8
if ($run029Pod.costPerHr -gt 1.01) {
 & $run029Cli pod stop $run029Pod.id
 throw 'Returned rate exceeds quoted GPU plus storage allowance; stopped for audit'
}
$run029Lease | ConvertTo-Json

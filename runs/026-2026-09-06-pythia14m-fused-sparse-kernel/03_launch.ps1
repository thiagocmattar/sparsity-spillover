param([string]$Attempt='rtx5090-001',[string]$Gpu='NVIDIA GeForce RTX 5090',[double]$QuotedRate=0.69,[double]$Hours=4)
$ErrorActionPreference='Stop'
$run026Root=(Resolve-Path "$PSScriptRoot/../..").Path
$run026Cli=Join-Path $run026Root 'tmp/runpodctl-v2.12.0.exe'
$run026Folder=Join-Path $PSScriptRoot "launch-control/$Attempt"
$run026Name="run026-$Attempt"
if (Test-Path -LiteralPath $run026Folder) { throw 'Attempt folder exists; inspect resources before retry' }
$null=New-Item -ItemType Directory -Path $run026Folder
$run026Start=[DateTimeOffset]::UtcNow
$run026Deadline=$run026Start.AddHours($Hours)
$run026Raw=& $run026Cli pod create --name $run026Name --image 'runpod/pytorch@sha256:0a360022e8de4375af99430f84e8b38951acc397252163a37ceac7204d01be35' --cloud-type COMMUNITY --gpu-id $Gpu --gpu-count 1 --min-cuda-version 12.8 --container-disk-in-gb 40 --volume-in-gb 40 --volume-mount-path /workspace --ports '22/tcp' --public-ip --ssh
if ($LASTEXITCODE -ne 0) { throw 'Create failed: list resources by exact name before any retry' }
$run026Pod=$run026Raw | ConvertFrom-Json
if (-not $run026Pod.id -or $run026Pod.name -ne $run026Name) { throw 'Unexpected response: inspect control plane' }
$run026Args=@('-NoProfile','-ExecutionPolicy','Bypass','-File',"`"$PSScriptRoot/02_stop_guard.ps1`"",'-PodId',$run026Pod.id,'-ExpectedName',$run026Name,'-DeadlineUtc',$run026Deadline.ToString('o'),'-RunpodctlPath',"`"$run026Cli`"",'-LogPath',"`"$run026Folder/guard.log`"")
$run026Guard=Start-Process powershell.exe -WindowStyle Hidden -ArgumentList $run026Args -PassThru -RedirectStandardError "$run026Folder/guard.err"
$run026Lease=[ordered]@{pod_id=$run026Pod.id;name=$run026Name;start_utc=$run026Start.ToString('o');stop_deadline_utc=$run026Deadline.ToString('o');quoted_rate=$QuotedRate;returned_rate=$run026Pod.costPerHr;max_hours=$Hours;guard_pid=$run026Guard.Id;total_run_budget_usd=25;workspace='/workspace/sparsity-spillover';image='runpod/pytorch@sha256:0a360022e8de4375af99430f84e8b38951acc397252163a37ceac7204d01be35'}
$run026Lease | ConvertTo-Json | Set-Content -LiteralPath "$run026Folder/lease.json" -Encoding UTF8
$run026Lease | ConvertTo-Json

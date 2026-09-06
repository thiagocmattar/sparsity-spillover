param([string]$Attempt='rtx5090-001',[string]$Gpu='NVIDIA GeForce RTX 5090',[double]$QuotedRate=0.69,[double]$Hours=4,[ValidateSet('COMMUNITY','SECURE')][string]$CloudType='COMMUNITY',[string]$Country='')
$ErrorActionPreference='Stop'
$run027Root=(Resolve-Path "$PSScriptRoot/../..").Path
$run027Cli=Join-Path $run027Root 'tmp/runpodctl-v2.12.0.exe'
$run027Folder=Join-Path $PSScriptRoot "launch-control/$Attempt"
$run027Name="run027-$Attempt"
if (Test-Path -LiteralPath $run027Folder) { throw 'Attempt folder exists; inspect resources before retry' }
$null=New-Item -ItemType Directory -Path $run027Folder
$run027Start=[DateTimeOffset]::UtcNow
$run027Deadline=$run027Start.AddHours($Hours)
$run027Placement=@()
if ($Country) { $run027Placement=@('--country-code',$Country) }
$run027Raw=& $run027Cli pod create --name $run027Name --image 'runpod/pytorch@sha256:0a360022e8de4375af99430f84e8b38951acc397252163a37ceac7204d01be35' --cloud-type $CloudType --gpu-id $Gpu --gpu-count 1 --min-cuda-version 12.8 --container-disk-in-gb 40 --volume-in-gb 40 --volume-mount-path /workspace --ports '22/tcp' --public-ip --ssh @run027Placement
if ($LASTEXITCODE -ne 0) { throw 'Create failed: list resources by exact name before any retry' }
$run027Pod=$run027Raw | ConvertFrom-Json
if (-not $run027Pod.id -or $run027Pod.name -ne $run027Name) { throw 'Unexpected response: inspect control plane' }
$run027Args=@('-NoProfile','-ExecutionPolicy','Bypass','-File',"`"$PSScriptRoot/06_stop_guard.ps1`"",'-PodId',$run027Pod.id,'-ExpectedName',$run027Name,'-DeadlineUtc',$run027Deadline.ToString('o'),'-RunpodctlPath',"`"$run027Cli`"",'-LogPath',"`"$run027Folder/guard.log`"")
$run027Guard=Start-Process powershell.exe -WindowStyle Hidden -ArgumentList $run027Args -PassThru -RedirectStandardError "$run027Folder/guard.err"
$run027Lease=[ordered]@{pod_id=$run027Pod.id;name=$run027Name;cloud_type=$CloudType;start_utc=$run027Start.ToString('o');stop_deadline_utc=$run027Deadline.ToString('o');quoted_rate=$QuotedRate;returned_rate=$run027Pod.costPerHr;max_hours=$Hours;guard_pid=$run027Guard.Id;total_run_budget_usd=5;workspace='/workspace/sparsity-spillover';image='runpod/pytorch@sha256:0a360022e8de4375af99430f84e8b38951acc397252163a37ceac7204d01be35'}
$run027Lease | ConvertTo-Json | Set-Content -LiteralPath "$run027Folder/lease.json" -Encoding UTF8
$run027Lease | ConvertTo-Json

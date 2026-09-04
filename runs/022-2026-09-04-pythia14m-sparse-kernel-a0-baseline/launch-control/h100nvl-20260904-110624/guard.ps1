$ErrorActionPreference = "Continue"

$podId = "n3ovkit8h3y2wh"
$deadline = [DateTimeOffset]::Parse("2026-09-04T17:06:46Z")
$runpodctl = "C:\Users\thima\Documents\Doutorado\exps\sparsity-spillover\tmp\runpodctl-v2.12.0.exe"
$logPath = "C:\Users\thima\Documents\Doutorado\exps\sparsity-spillover\runs\022-2026-09-04-pythia14m-sparse-kernel-a0-baseline\launch-control\h100nvl-20260904-110624\guard.log"

$remainingSeconds = [Math]::Max(0, [Math]::Ceiling(($deadline - [DateTimeOffset]::UtcNow).TotalSeconds))
"guard_started_utc=$([DateTimeOffset]::UtcNow.ToString('o')) pod_id=$podId deadline_utc=$($deadline.ToString('o')) sleep_seconds=$remainingSeconds" | Add-Content -LiteralPath $logPath
Start-Sleep -Seconds $remainingSeconds
"guard_fired_utc=$([DateTimeOffset]::UtcNow.ToString('o')) pod_id=$podId" | Add-Content -LiteralPath $logPath
& $runpodctl pod delete $podId 2>&1 | Add-Content -LiteralPath $logPath
"guard_finished_utc=$([DateTimeOffset]::UtcNow.ToString('o')) pod_id=$podId exit_code=$LASTEXITCODE" | Add-Content -LiteralPath $logPath

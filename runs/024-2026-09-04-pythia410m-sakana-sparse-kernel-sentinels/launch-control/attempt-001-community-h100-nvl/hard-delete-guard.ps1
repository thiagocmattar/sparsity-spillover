$ErrorActionPreference = "Stop"

$podId = "q10a8y77vkkavg"
$deadline = [DateTimeOffset]::Parse("2026-09-05T04:52:25.660Z")
$runpodctl = "C:\Users\thima\Documents\Doutorado\exps\sparsity-spillover\tmp\runpodctl-v2.12.0.exe"
$log = "C:\Users\thima\Documents\Doutorado\exps\sparsity-spillover\runs\024-2026-09-04-pythia410m-sakana-sparse-kernel-sentinels\launch-control\attempt-001-community-h100-nvl\hard-delete-guard.log"

"guard_started_utc=$([DateTimeOffset]::UtcNow.ToString('o'))" | Out-File -LiteralPath $log -Encoding utf8
"pod_id=$podId" | Out-File -LiteralPath $log -Encoding utf8 -Append
"deadline_utc=$($deadline.ToString('o'))" | Out-File -LiteralPath $log -Encoding utf8 -Append
$remaining = [Math]::Ceiling(($deadline - [DateTimeOffset]::UtcNow).TotalSeconds)
if ($remaining -gt 0) {
    Start-Sleep -Seconds $remaining
}
"guard_woke_utc=$([DateTimeOffset]::UtcNow.ToString('o'))" | Out-File -LiteralPath $log -Encoding utf8 -Append
& $runpodctl pod get $podId 2>&1 | Out-File -LiteralPath $log -Encoding utf8 -Append
if ($LASTEXITCODE -eq 0) {
    & $runpodctl pod delete $podId 2>&1 | Out-File -LiteralPath $log -Encoding utf8 -Append
    "delete_exit_code=$LASTEXITCODE" | Out-File -LiteralPath $log -Encoding utf8 -Append
} else {
    "pod_absent_at_deadline=true" | Out-File -LiteralPath $log -Encoding utf8 -Append
}

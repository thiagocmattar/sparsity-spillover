param([ValidateRange(0, 300)][int]$InitialDelaySeconds = 0)
$ErrorActionPreference = 'Stop'
$run032Root = Split-Path -Parent $PSScriptRoot
$run032Repo = Split-Path -Parent $run032Root
$run032Python = Join-Path $run032Repo '.venv/Scripts/python.exe'
Set-Location -LiteralPath $run032Repo
if ($InitialDelaySeconds -gt 0) { Start-Sleep -Seconds $InitialDelaySeconds }
while ($true) {
    & $run032Python (Join-Path $PSScriptRoot '22_retry_kappa05.py')
    if ($LASTEXITCODE -eq 0) {
        & $run032Python (Join-Path $PSScriptRoot '03_verify.py')
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
        & $run032Python (Join-Path $run032Repo 'analyses/022-2026-09-16-a7-h-only-pressure/01_compare.py')
        exit $LASTEXITCODE
    }
    if ($LASTEXITCODE -ne 75) { exit $LASTEXITCODE }
    Start-Sleep -Seconds 300
}

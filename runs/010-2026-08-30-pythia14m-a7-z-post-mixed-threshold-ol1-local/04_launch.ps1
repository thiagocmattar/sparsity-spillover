param(
    [string]$Python = 'C:\Users\thima\AppData\Local\Programs\Python\Python312\python.exe'
)

$ErrorActionPreference = 'Stop'
$runDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = (Resolve-Path -LiteralPath (Join-Path $runDir '..\..')).Path
$artifacts = Join-Path $runDir 'artifacts'
$attempts = Join-Path $artifacts 'attempts'
$processIdPath = Join-Path $artifacts 'cohort.pid'
$launchPlanPath = Join-Path $runDir 'prelaunch\launch-plan.json'

if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "CUDA Python interpreter not found: $Python"
}
if (-not (Test-Path -LiteralPath $launchPlanPath -PathType Leaf)) {
    throw 'Run 010 launch plan is missing.'
}
$launchPlan = Get-Content -Raw -LiteralPath $launchPlanPath | ConvertFrom-Json
if ($launchPlan.launch_approved -ne $true) {
    throw 'Run 010 does not have explicit launch approval.'
}
if (Test-Path -LiteralPath $attempts) {
    $existing = Get-ChildItem -LiteralPath $attempts -Directory
    if ($existing.Count -gt 0) {
        throw 'Run 010 already has scientific attempts; refusing to relaunch.'
    }
}
if (Test-Path -LiteralPath $processIdPath) {
    $priorProcessId = [int](Get-Content -Raw -LiteralPath $processIdPath)
    if (Get-Process -Id $priorProcessId -ErrorAction SilentlyContinue) {
        throw "Run 010 cohort process $priorProcessId is already active."
    }
}

$gitCommit = (& git -C $repoRoot rev-parse HEAD).Trim()
if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($gitCommit)) {
    throw 'Could not record the launch Git commit.'
}
$gitStatus = & git -C $repoRoot status --porcelain
if ($LASTEXITCODE -ne 0) {
    throw 'Could not record the launch Git dirty state.'
}
$gitDirty = -not [string]::IsNullOrWhiteSpace(($gitStatus -join "`n"))

New-Item -ItemType Directory -Path $artifacts -Force | Out-Null
$provenance = [ordered]@{
    schema_version = 1
    recorded_at = (Get-Date).ToUniversalTime().ToString('o')
    git_commit = $gitCommit
    git_dirty = $gitDirty
    config_sha256 = $launchPlan.config_sha256
    run_code_content_sha256 = $launchPlan.run_code_content_sha256
    training_schedule_sha256 = $launchPlan.training_schedule_sha256
    calibration = $launchPlan.calibration
}
$provenance | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $artifacts 'launch-provenance.json') -Encoding ascii

$stdout = Join-Path $artifacts 'cohort.stdout.log'
$stderr = Join-Path $artifacts 'cohort.stderr.log'
$entrypoint = Join-Path $runDir '02_train.py'
$detachedLauncher = Join-Path $runDir 'launch_detached.py'
$launchedProcessId = & $Python $detachedLauncher $Python $entrypoint $repoRoot $stdout $stderr
if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($launchedProcessId)) {
    throw 'Detached Python launcher failed.'
}
[int]$launchedProcessId | Set-Content -LiteralPath $processIdPath -Encoding ascii
[int]$launchedProcessId



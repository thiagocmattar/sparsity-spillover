param(
    [string]$Python = 'C:\Users\thima\AppData\Local\Programs\Python\Python312\python.exe'
)

$ErrorActionPreference = 'Stop'
$runDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = (Resolve-Path -LiteralPath (Join-Path $runDir '..\..')).Path
$artifacts = Join-Path $runDir 'artifacts'
$attempts = Join-Path $artifacts 'attempts'
$processIdPath = Join-Path $artifacts 'cohort.pid'

if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "CUDA Python interpreter not found: $Python"
}
if (Test-Path -LiteralPath $attempts) {
    $existing = Get-ChildItem -LiteralPath $attempts -Directory
    if ($existing.Count -gt 0) {
        throw 'Run 005 already has scientific attempts; refusing to relaunch.'
    }
}
if (Test-Path -LiteralPath $processIdPath) {
    $priorProcessId = [int](Get-Content -Raw -LiteralPath $processIdPath)
    if (Get-Process -Id $priorProcessId -ErrorAction SilentlyContinue) {
        throw "Run 005 cohort process $priorProcessId is already active."
    }
}
New-Item -ItemType Directory -Path $artifacts -Force | Out-Null
$stdout = Join-Path $artifacts 'cohort.stdout.log'
$stderr = Join-Path $artifacts 'cohort.stderr.log'
$entrypoint = Join-Path $runDir '02_train.py'
$process = Start-Process `
    -FilePath $Python `
    -ArgumentList @($entrypoint) `
    -WorkingDirectory $repoRoot `
    -RedirectStandardOutput $stdout `
    -RedirectStandardError $stderr `
    -WindowStyle Hidden `
    -PassThru
$process.Id | Set-Content -LiteralPath $processIdPath -Encoding ascii
$process.Id

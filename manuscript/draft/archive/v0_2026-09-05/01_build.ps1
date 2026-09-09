$ErrorActionPreference = 'Stop'
$DraftDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = (Resolve-Path (Join-Path $DraftDir '..\..')).Path
$AnalysisDir = Join-Path $RepoRoot 'analyses\013-2026-09-04-matched-intervention-manuscript'
$Python = Join-Path $RepoRoot '.venv\Scripts\python.exe'

& $Python (Join-Path $AnalysisDir '01_build.py')
if ($LASTEXITCODE -ne 0) { throw "Evidence build failed: $LASTEXITCODE" }

$Manifest = [ordered]@{}
foreach ($Kind in @('figures', 'tables')) {
    $DestinationDir = Join-Path $DraftDir $Kind
    New-Item -ItemType Directory -Force -Path $DestinationDir | Out-Null
    $Extension = if ($Kind -eq 'figures') { '*.pdf' } else { '*.tex' }
    foreach ($Source in Get-ChildItem -LiteralPath (Join-Path $AnalysisDir $Kind) -Filter $Extension -File) {
        $Relative = "$Kind/$($Source.Name)"
        Copy-Item -LiteralPath $Source.FullName -Destination (Join-Path $DraftDir $Relative) -Force
        $Manifest[$Relative] = (Get-FileHash -Algorithm SHA256 -LiteralPath $Source.FullName).Hash.ToLowerInvariant()
    }
}
$Manifest | ConvertTo-Json | Set-Content -Encoding utf8 (Join-Path $DraftDir 'build-input-hashes.json')
Push-Location $DraftDir
try {
    & latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
    if ($LASTEXITCODE -ne 0) { throw "latexmk failed: $LASTEXITCODE" }
} finally {
    Pop-Location
}
Write-Host "Built $DraftDir\main.pdf"

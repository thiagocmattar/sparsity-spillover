$ErrorActionPreference = 'Stop'

$DraftDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = (Resolve-Path (Join-Path $DraftDir '..\..')).Path
$FigureDir = Join-Path $DraftDir 'figures'
$TableDir = Join-Path $DraftDir 'tables'

New-Item -ItemType Directory -Force -Path $FigureDir, $TableDir | Out-Null

$Copies = @(
    @('analyses\012-2026-09-04-paper-synthesis\figures\01-absolute-frontiers-by-scale.pdf', 'figures\01-absolute-frontiers-by-scale.pdf'),
    @('analyses\012-2026-09-04-paper-synthesis\figures\02-within-scale-deltas.pdf', 'figures\02-within-scale-deltas.pdf'),
    @('analyses\012-2026-09-04-paper-synthesis\figures\03-sitewise-zero-structure.pdf', 'figures\03-sitewise-zero-structure.pdf'),
    @('analyses\012-2026-09-04-paper-synthesis\figures\04-operation-contributions.pdf', 'figures\04-operation-contributions.pdf'),
    @('analyses\011-2026-09-03-pythia14m-70m-410m-selected-ladder\figures\03-a0-gradient-norm-vs-tokens.pdf', 'figures\05-a0-gradient-norm-vs-tokens.pdf'),
    @('runs\004-2026-08-29-pythia14m-full-pass-l1n\figures\01-h-vs-site-near-zero-grid.pdf', 'figures\06-spillover-site-grid.pdf'),
    @('runs\004-2026-08-29-pythia14m-full-pass-l1n\figures\02-h-vs-attention-output-near-zero.pdf', 'figures\07-attention-output-near-zero.pdf'),
    @('analyses\003-2026-08-30-run004-vs-run009-full-pass-l1-ol1\figures\03-ol1-orthogonalization-trajectories.pdf', 'figures\08-ol1-orthogonalization-trajectories.pdf'),
    @('analyses\012-2026-09-04-paper-synthesis\tables\baseline-exposure.tex', 'tables\baseline-exposure.tex'),
    @('analyses\012-2026-09-04-paper-synthesis\tables\cross-scale-endpoints.tex', 'tables\cross-scale-endpoints.tex')
)

$Manifest = [ordered]@{}
foreach ($Pair in $Copies) {
    $Source = Join-Path $RepoRoot $Pair[0]
    $Destination = Join-Path $DraftDir $Pair[1]
    if (-not (Test-Path -LiteralPath $Source)) {
        throw "Missing paper input: $Source"
    }
    Copy-Item -LiteralPath $Source -Destination $Destination -Force
    $Manifest[$Pair[1]] = (Get-FileHash -Algorithm SHA256 -LiteralPath $Source).Hash.ToLowerInvariant()
}

$Manifest | ConvertTo-Json | Set-Content -Encoding utf8 (Join-Path $DraftDir 'build-input-hashes.json')

Push-Location $DraftDir
try {
    & latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
    if ($LASTEXITCODE -ne 0) {
        throw "latexmk failed with exit code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}

Write-Host "Built $DraftDir\main.pdf"


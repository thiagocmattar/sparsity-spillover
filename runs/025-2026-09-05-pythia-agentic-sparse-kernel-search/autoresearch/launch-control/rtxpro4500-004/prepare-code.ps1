$ErrorActionPreference = 'Stop'
$run025Run = (Resolve-Path "$PSScriptRoot/../../..").Path
$run025Root = (Resolve-Path "$run025Run/../..").Path
$run025Bundle = Join-Path $run025Run 'autoresearch/bundles/phase17-code-001'
$run025Archive = Join-Path $run025Bundle 'run025-phase17-code.tar.gz'
$run025Record = Join-Path $run025Bundle 'archive.json'
$run025Verification = Join-Path $run025Bundle 'source-verification.json'
$run025Sums = Join-Path $run025Bundle 'SHA256SUMS'
if (Test-Path -LiteralPath $run025Bundle) {
    throw 'Phase 17 bundle already exists; do not overwrite a transfer identity'
}
New-Item -ItemType Directory -Path $run025Bundle | Out-Null
$run025Commit = (& git -C $run025Root rev-parse HEAD).Trim()
if ($LASTEXITCODE -ne 0 -or $run025Commit -notmatch '^[0-9a-f]{40}$') {
    throw 'Unable to resolve the committed source identity'
}
& git -C $run025Root -c core.autocrlf=false archive --format=tar.gz `
    --output=$run025Archive $run025Commit -- `
    src runs/025-2026-09-05-pythia-agentic-sparse-kernel-search
if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $run025Archive)) {
    throw 'LF-preserving git archive failed'
}
$run025Verifier = Join-Path $PSScriptRoot 'verify_code_bundle.py'
$run025VerifiedText = & python $run025Verifier $run025Archive
if ($LASTEXITCODE -ne 0) { throw 'Phase 17 source verification failed' }
[IO.File]::WriteAllText($run025Verification, ($run025VerifiedText -join "`n") + "`n", [Text.Encoding]::UTF8)
$run025Hash = (Get-FileHash -LiteralPath $run025Archive -Algorithm SHA256).Hash.ToLowerInvariant()
$run025Bytes = (Get-Item -LiteralPath $run025Archive).Length
$run025Payload = [ordered]@{
    created_utc = [DateTimeOffset]::UtcNow.ToString('o')
    git_commit = $run025Commit
    archive_core_autocrlf = $false
    path = $run025Archive
    remote_name = 'run025-phase17-code.tar.gz'
    expected_bytes = $run025Bytes
    expected_sha256 = $run025Hash
    persistent_inputs = 'verified Run 025 network volume; no checkpoint or token-cache transfer'
}
$run025Payload | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $run025Record -Encoding UTF8
[IO.File]::WriteAllText($run025Sums, "$run025Hash  run025-phase17-code.tar.gz`n", [Text.Encoding]::ASCII)
$run025Payload | ConvertTo-Json -Depth 4

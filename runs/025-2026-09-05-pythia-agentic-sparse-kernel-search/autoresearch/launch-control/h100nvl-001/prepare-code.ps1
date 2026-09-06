$ErrorActionPreference = 'Stop'
$run025Run = (Resolve-Path "$PSScriptRoot/../../..").Path
$run025Root = (Resolve-Path "$run025Run/../..").Path
$run025Bundle = Join-Path $run025Run 'autoresearch/bundles/h100-code-005'
$run025Archive = Join-Path $run025Bundle 'run025-h100-code.tar.gz'
$run025Record = Join-Path $run025Bundle 'archive.json'
$run025Sums = Join-Path $run025Bundle 'SHA256SUMS'
if (Test-Path -LiteralPath $run025Bundle) {
    throw 'H100 code bundle already exists; do not overwrite a transfer identity'
}
New-Item -ItemType Directory -Path $run025Bundle | Out-Null
$run025Commit = (& git -C $run025Root rev-parse HEAD).Trim()
if ($LASTEXITCODE -ne 0 -or $run025Commit -notmatch '^[0-9a-f]{40}$') {
    throw 'Unable to resolve the committed source identity'
}
& git -C $run025Root archive --format=tar.gz --output=$run025Archive $run025Commit -- `
    src runs/025-2026-09-05-pythia-agentic-sparse-kernel-search
if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $run025Archive)) {
    throw 'git archive failed'
}
$run025Inputs = @(
    [ordered]@{
        path = (Join-Path $run025Run 'autoresearch/bundles/inputs-14m-001/payload.tar.gz')
        remote_name = 'payload.tar.gz'
        expected_bytes = 313922753
        expected_sha256 = 'a0867d2211b0a769b990a19fa78b154be4b087842d2c3cbf37304d7dc8767a39'
    },
    [ordered]@{
        path = (Join-Path $run025Run 'prelaunch/run025-70m-development.tar')
        remote_name = 'run025-70m-development.tar'
        expected_bytes = 1690552320
        expected_sha256 = 'fa74f92130fd511f4f98206cf2ca9d2446afa7fe8ca9b20ea5d21a3f394deb6c'
    },
    [ordered]@{
        path = (Join-Path $run025Run 'prelaunch/run025-410m-development.tar')
        remote_name = 'run025-410m-development.tar'
        expected_bytes = 9728491520
        expected_sha256 = '4d7168518eb307e7e05d963f6efbac7ab4091ba7db17d56bbbfcb02a8bae79fd'
    }
)
foreach ($run025Input in $run025Inputs) {
    if ((Get-Item -LiteralPath $run025Input.path).Length -ne $run025Input.expected_bytes) {
        throw "Input byte-count mismatch: $($run025Input.remote_name)"
    }
    $run025Actual = (Get-FileHash -LiteralPath $run025Input.path -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($run025Actual -ne $run025Input.expected_sha256) {
        throw "Input SHA-256 mismatch: $($run025Input.remote_name)"
    }
}
$run025Code = [ordered]@{
    path = $run025Archive
    remote_name = 'run025-h100-code.tar.gz'
    expected_bytes = (Get-Item -LiteralPath $run025Archive).Length
    expected_sha256 = (Get-FileHash -LiteralPath $run025Archive -Algorithm SHA256).Hash.ToLowerInvariant()
}
$run025Items = @($run025Inputs) + @($run025Code)
$run025TotalBytes = [long]0
foreach ($run025Item in $run025Items) {
    $run025TotalBytes += [long]$run025Item.expected_bytes
}
$run025Payload = [ordered]@{
    created_utc = [DateTimeOffset]::UtcNow.ToString('o')
    git_commit = $run025Commit
    extraction_order = @('payload.tar.gz', 'run025-70m-development.tar', 'run025-410m-development.tar', 'run025-h100-code.tar.gz')
    items = $run025Items
    total_bytes = $run025TotalBytes
}
$run025Payload | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $run025Record -Encoding UTF8
$run025SumLines = @($run025Items | ForEach-Object { "$($_.expected_sha256)  $($_.remote_name)" })
$run025Ascii = [Text.Encoding]::ASCII
[IO.File]::WriteAllText($run025Sums, ([string]::Join("`n", $run025SumLines) + "`n"), $run025Ascii)
$run025Payload | ConvertTo-Json -Depth 5

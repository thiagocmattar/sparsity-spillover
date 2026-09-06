$ErrorActionPreference = 'Stop'
$run025Run = (Resolve-Path "$PSScriptRoot/../../..").Path
$run025Root = (Resolve-Path "$run025Run/../..").Path
$run025Cli = Join-Path $run025Root 'tmp/runpodctl-v2.12.0.exe'
$run025Bundle = Join-Path $run025Run 'autoresearch/bundles/h100-code-006'
$run025Manifest = Get-Content -Raw (Join-Path $run025Bundle 'archive.json') | ConvertFrom-Json
$run025Lease = Get-Content -Raw (Join-Path $PSScriptRoot 'lease.json') | ConvertFrom-Json
$run025StartRecord = Join-Path $PSScriptRoot 'start.json'
if (Test-Path -LiteralPath $run025StartRecord) {
    throw 'Start record already exists; inspect the scoped worker instead of relaunching'
}
$run025Target = (& $run025Cli pod get $run025Lease.pod_id 2>$null) | ConvertFrom-Json
if ($LASTEXITCODE -ne 0 -or $run025Target.name -ne $run025Lease.name) {
    throw 'Scoped H100 Pod is unavailable or has the wrong name'
}
$run025Ssh = (& $run025Cli ssh info $run025Lease.pod_id 2>$null) | ConvertFrom-Json
if ($LASTEXITCODE -ne 0 -or -not $run025Ssh.ip -or -not $run025Ssh.port) {
    throw 'Scoped H100 Pod is not SSH-ready'
}
$run025KnownHosts = Join-Path $PSScriptRoot 'known_hosts'
if (-not (Test-Path -LiteralPath $run025KnownHosts) -or (Get-Item -LiteralPath $run025KnownHosts).Length -eq 0) {
    throw 'Attempt-scoped SSH host key bootstrap is missing'
}
$run025SshArgs = @(
    '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=20',
    '-o', 'StrictHostKeyChecking=yes', '-o', "UserKnownHostsFile=$run025KnownHosts",
    '-i', $run025Ssh.ssh_key.path, '-p', [string]$run025Ssh.port,
    ('root@' + $run025Ssh.ip)
)
$run025RemoteInstall = @'
set -euo pipefail
BASE=/workspace/run025-h100nvl-001
mkdir -p "$BASE/incoming" "$BASE/bootstrap-control"
curl -fL --max-time 300 -o "$BASE/runpodctl-v2.12.0.download" https://github.com/runpod/runpodctl/releases/download/v2.12.0/runpodctl-linux-amd64
printf '%s  %s\n' f273555b935963925e696e95f36a883ca68c5c845efc893db9f8f701749c8474 "$BASE/runpodctl-v2.12.0.download" | sha256sum -c -
install -m 0755 "$BASE/runpodctl-v2.12.0.download" "$BASE/runpodctl-v2.12.0"
'@
& ssh.exe @run025SshArgs $run025RemoteInstall
if ($LASTEXITCODE -ne 0) { throw 'Remote transfer helper installation failed' }

$run025Bootstrap = Join-Path $PSScriptRoot '../h100nvl-001/bootstrap-and-run.sh'
$run025Sums = Join-Path $run025Bundle 'SHA256SUMS'
& scp.exe -q -o BatchMode=yes -o StrictHostKeyChecking=yes -o "UserKnownHostsFile=$run025KnownHosts" `
    -i $run025Ssh.ssh_key.path -P $run025Ssh.port `
    $run025Bootstrap $run025Sums ("root@$($run025Ssh.ip):/workspace/run025-h100nvl-001/")
if ($LASTEXITCODE -ne 0) { throw 'Bootstrap metadata transfer failed' }
& ssh.exe @run025SshArgs 'mv /workspace/run025-h100nvl-001/SHA256SUMS /workspace/run025-h100nvl-001/incoming/SHA256SUMS && chmod 0755 /workspace/run025-h100nvl-001/bootstrap-and-run.sh'
if ($LASTEXITCODE -ne 0) { throw 'Remote bootstrap staging failed' }

$run025Scratch = Join-Path $run025Root 'tmp/run025-h100nvl-002-transfer'
New-Item -ItemType Directory -Path $run025Scratch -ErrorAction Stop | Out-Null
$run025Transfers = @()
foreach ($run025Item in $run025Manifest.items) {
    $run025Path = [string]$run025Item.path
    if ((Get-Item -LiteralPath $run025Path).Length -ne [long]$run025Item.expected_bytes) {
        throw "Transfer byte-count mismatch: $($run025Item.remote_name)"
    }
    if ((Get-FileHash -LiteralPath $run025Path -Algorithm SHA256).Hash.ToLowerInvariant() -ne $run025Item.expected_sha256) {
        throw "Transfer hash mismatch: $($run025Item.remote_name)"
    }
    $run025Tag = [IO.Path]::GetFileNameWithoutExtension([string]$run025Item.remote_name) -replace '[^A-Za-z0-9-]', '-'
    $run025RequestedCode = 'run025-' + [Guid]::NewGuid().ToString('N')
    $run025SendOut = Join-Path $run025Scratch "$run025Tag-send.out"
    $run025SendErr = Join-Path $run025Scratch "$run025Tag-send.err"
    $run025ReceiveOut = Join-Path $run025Scratch "$run025Tag-receive.out"
    $run025ReceiveErr = Join-Path $run025Scratch "$run025Tag-receive.err"
    $run025Send = Start-Process -FilePath $run025Cli -WindowStyle Hidden `
        -ArgumentList @('send', "`"$run025Path`"", '--code', $run025RequestedCode) `
        -PassThru -RedirectStandardOutput $run025SendOut -RedirectStandardError $run025SendErr
    $run025ActualCode = $null
    $run025CodeDeadline = [DateTimeOffset]::UtcNow.AddSeconds(45)
    while (-not $run025ActualCode -and [DateTimeOffset]::UtcNow -lt $run025CodeDeadline) {
        Start-Sleep -Milliseconds 250
        $run025Text = (Get-Content -LiteralPath $run025SendOut -Raw -ErrorAction SilentlyContinue) + `
            (Get-Content -LiteralPath $run025SendErr -Raw -ErrorAction SilentlyContinue)
        $run025Match = [regex]::Match($run025Text, 'code is:\s*(\S+)\r?\n')
        if ($run025Match.Success) { $run025ActualCode = $run025Match.Groups[1].Value }
        if ($run025Send.HasExited -and -not $run025ActualCode) {
            throw 'Relay sender exited before producing a complete code line'
        }
    }
    if (-not $run025ActualCode -or $run025ActualCode -notmatch '^[A-Za-z0-9-]+$') {
        Stop-Process -Id $run025Send.Id -ErrorAction SilentlyContinue
        throw 'Relay sender did not provide a valid ephemeral code'
    }
    $run025RemoteReceive = "cd /workspace/run025-h100nvl-001/incoming && /workspace/run025-h100nvl-001/runpodctl-v2.12.0 receive $run025ActualCode"
    $run025Receive = Start-Process -FilePath ssh.exe -WindowStyle Hidden `
        -ArgumentList @($run025SshArgs + @("`"$run025RemoteReceive`"")) `
        -PassThru -RedirectStandardOutput $run025ReceiveOut -RedirectStandardError $run025ReceiveErr
    $run025Started = [DateTimeOffset]::UtcNow
    $run025Deadline = $run025Started.AddMinutes(30)
    while ((-not $run025Send.HasExited -or -not $run025Receive.HasExited) -and [DateTimeOffset]::UtcNow -lt $run025Deadline) {
        Start-Sleep -Seconds 10
    }
    if (-not $run025Send.HasExited -or -not $run025Receive.HasExited) {
        Stop-Process -Id $run025Send.Id -ErrorAction SilentlyContinue
        Stop-Process -Id $run025Receive.Id -ErrorAction SilentlyContinue
        throw "Relay transfer deadline exceeded: $($run025Item.remote_name)"
    }
    $run025Send.WaitForExit()
    $run025Receive.WaitForExit()
    $run025Finished = [DateTimeOffset]::UtcNow
    $run025Transfers += [ordered]@{
        remote_name = $run025Item.remote_name
        bytes = [long]$run025Item.expected_bytes
        sha256 = $run025Item.expected_sha256
        started_utc = $run025Started.ToString('o')
        finished_utc = $run025Finished.ToString('o')
        elapsed_seconds = [Math]::Round(($run025Finished - $run025Started).TotalSeconds, 1)
        sender_exited = $run025Send.HasExited
        receiver_exited = $run025Receive.HasExited
    }
}

$run025RemoteStart = @'
set -euo pipefail
cd /workspace/run025-h100nvl-001/incoming
sha256sum -c SHA256SUMS
cd /workspace/run025-h100nvl-001
nohup setsid env PATH="/usr/local/cuda/bin:$PATH" bash bootstrap-and-run.sh > bootstrap-control/outer.log 2>&1 < /dev/null &
printf '%s\n' "$!"
'@
$run025Pid = (& ssh.exe @run025SshArgs $run025RemoteStart | Select-Object -Last 1).Trim()
if ($LASTEXITCODE -ne 0 -or $run025Pid -notmatch '^\d+$') {
    throw 'Remote bootstrap did not return a process id'
}
$run025Start = [ordered]@{
    started_utc = [DateTimeOffset]::UtcNow.ToString('o')
    pod_id = $run025Lease.pod_id
    source_git_commit = $run025Manifest.git_commit
    total_transfer_bytes = [long]$run025Manifest.total_bytes
    transfers = $run025Transfers
    outer_pid = [int]$run025Pid
    route = 'RunPod encrypted one-time-code relay; ephemeral codes retained only in untracked temporary logs'
}
$run025Start | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $run025StartRecord -Encoding UTF8
$run025Start | ConvertTo-Json -Depth 6

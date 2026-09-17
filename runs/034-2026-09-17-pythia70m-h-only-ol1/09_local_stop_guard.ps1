param(
 [Parameter(Mandatory=$true)][string]$PodId,
 [Parameter(Mandatory=$true)][string]$ExpectedName,
 [Parameter(Mandatory=$true)][DateTimeOffset]$DeadlineUtc,
 [Parameter(Mandatory=$true)][string]$RunpodctlPath,
 [Parameter(Mandatory=$true)][string]$LogPath
)
$ErrorActionPreference = 'Stop'
if ($PodId -notmatch '^[a-z0-9]+$' -or $ExpectedName -notlike 'run034-*') { throw 'Invalid scoped identity' }
$run034Cli = (Resolve-Path -LiteralPath $RunpodctlPath).Path
function Read-Target {
 $run034Raw = & $run034Cli pod get $PodId
 if ($LASTEXITCODE -ne 0) { throw 'Pod lookup failed' }
 $run034Target = $run034Raw | ConvertFrom-Json
 if ($run034Target.name -ne $ExpectedName) { throw 'Name mismatch' }
 return $run034Target
}
$null = Read-Target
Add-Content -LiteralPath $LogPath -Value "ARMED $PodId deadline $($DeadlineUtc.ToString('o'))"
while ([DateTimeOffset]::UtcNow -lt $DeadlineUtc) {
 $run034Remaining = ($DeadlineUtc - [DateTimeOffset]::UtcNow).TotalSeconds
 Start-Sleep -Seconds ([Math]::Min(30, [Math]::Max(1, [Math]::Ceiling($run034Remaining))))
}
for ($run034Try=1; $run034Try -le 5; $run034Try++) {
 try {
  $null = Read-Target
  & $run034Cli pod stop $PodId
  if ($LASTEXITCODE -ne 0) { throw 'Stop failed' }
  $run034After = Read-Target
  if ($run034After.desiredStatus -notin @('EXITED','STOPPED')) { throw 'Stop unconfirmed' }
  Add-Content -LiteralPath $LogPath -Value "STOP_CONFIRMED $PodId; volume retained for retrieval"
  exit 0
 } catch {
  Add-Content -LiteralPath $LogPath -Value "RETRY $run034Try stop $PodId"
  Start-Sleep -Seconds 15
 }
}
exit 1

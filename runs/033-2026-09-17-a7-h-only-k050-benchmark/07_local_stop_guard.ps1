param(
 [Parameter(Mandatory=$true)][string]$PodId,
 [Parameter(Mandatory=$true)][string]$ExpectedName,
 [Parameter(Mandatory=$true)][DateTimeOffset]$DeadlineUtc,
 [Parameter(Mandatory=$true)][string]$RunpodctlPath,
 [Parameter(Mandatory=$true)][string]$LogPath
)
$ErrorActionPreference = 'Stop'
if ($PodId -notmatch '^[a-z0-9]+$' -or $ExpectedName -notlike 'run033-*') { throw 'Invalid scoped identity' }
$run033Cli = (Resolve-Path -LiteralPath $RunpodctlPath).Path
function Read-Target {
 $run033Raw = & $run033Cli pod get $PodId
 if ($LASTEXITCODE -ne 0) { throw 'Pod lookup failed' }
 $run033Target = $run033Raw | ConvertFrom-Json
 if ($run033Target.name -ne $ExpectedName) { throw 'Name mismatch' }
 return $run033Target
}
$null = Read-Target
Add-Content -LiteralPath $LogPath -Value "ARMED $PodId deadline $($DeadlineUtc.ToString('o'))"
while ([DateTimeOffset]::UtcNow -lt $DeadlineUtc) {
 $run033Remaining = ($DeadlineUtc - [DateTimeOffset]::UtcNow).TotalSeconds
 Start-Sleep -Seconds ([Math]::Min(30, [Math]::Max(1, [Math]::Ceiling($run033Remaining))))
}
for ($run033Try=1; $run033Try -le 5; $run033Try++) {
 try {
  $null = Read-Target
  & $run033Cli pod stop $PodId
  if ($LASTEXITCODE -ne 0) { throw 'Stop failed' }
  $run033After = Read-Target
  if ($run033After.desiredStatus -notin @('EXITED','STOPPED')) { throw 'Stop unconfirmed' }
  Add-Content -LiteralPath $LogPath -Value "STOP_CONFIRMED $PodId; volume retained for retrieval"
  exit 0
 } catch {
  Add-Content -LiteralPath $LogPath -Value "RETRY $run033Try stop $PodId"
  Start-Sleep -Seconds 15
 }
}
exit 1

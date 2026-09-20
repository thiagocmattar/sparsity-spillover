param(
 [Parameter(Mandatory=$true)][string]$PodId,
 [Parameter(Mandatory=$true)][string]$ExpectedName,
 [Parameter(Mandatory=$true)][DateTimeOffset]$DeadlineUtc,
 [Parameter(Mandatory=$true)][string]$RunpodctlPath,
 [Parameter(Mandatory=$true)][string]$LogPath
)
$ErrorActionPreference = 'Stop'
if ($PodId -notmatch '^[a-z0-9]+$' -or $ExpectedName -notlike 'run040-*') { throw 'Invalid scoped identity' }
$run040Cli = (Resolve-Path -LiteralPath $RunpodctlPath).Path
function Read-Target {
 $run040Raw = & $run040Cli pod get $PodId
 if ($LASTEXITCODE -ne 0) { throw 'Pod lookup failed' }
 $run040Target = $run040Raw | ConvertFrom-Json
 if ($run040Target.name -ne $ExpectedName) { throw 'Name mismatch' }
 return $run040Target
}
$null = Read-Target
Add-Content -LiteralPath $LogPath -Value "ARMED $PodId deadline $($DeadlineUtc.ToString('o'))"
while ([DateTimeOffset]::UtcNow -lt $DeadlineUtc) {
 $run040Remaining = ($DeadlineUtc - [DateTimeOffset]::UtcNow).TotalSeconds
 Start-Sleep -Seconds ([Math]::Min(30, [Math]::Max(1, [Math]::Ceiling($run040Remaining))))
}
for ($run040Try=1; $run040Try -le 5; $run040Try++) {
 try {
  $null = Read-Target
  & $run040Cli pod stop $PodId
  if ($LASTEXITCODE -ne 0) { throw 'Stop failed' }
  $run040After = Read-Target
  if ($run040After.status -notin @('EXITED','STOPPED') -and
      $run040After.desiredStatus -notin @('EXITED','STOPPED')) { throw 'Stop unconfirmed' }
  Add-Content -LiteralPath $LogPath -Value "STOP_CONFIRMED $PodId; volume retained for retrieval"
  exit 0
 } catch {
  Add-Content -LiteralPath $LogPath -Value "RETRY $run040Try stop $PodId"
  Start-Sleep -Seconds 15
 }
}
exit 1

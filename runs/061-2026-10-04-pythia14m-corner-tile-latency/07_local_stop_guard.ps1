param(
 [Parameter(Mandatory=$true)][string]$PodId,
 [Parameter(Mandatory=$true)][string]$ExpectedName,
 [Parameter(Mandatory=$true)][DateTimeOffset]$DeadlineUtc,
 [Parameter(Mandatory=$true)][string]$RunpodctlPath,
 [Parameter(Mandatory=$true)][string]$LogPath
)
$ErrorActionPreference = 'Stop'
if ($PodId -notmatch '^[a-z0-9]+$' -or $ExpectedName -notlike 'run061-*') { throw 'Invalid scoped identity' }
$run061Cli = (Resolve-Path -LiteralPath $RunpodctlPath).Path
function Read-Target {
 $run061Raw = & $run061Cli pod get $PodId
 if ($LASTEXITCODE -ne 0) { throw 'Pod lookup failed' }
 $run061Target = $run061Raw | ConvertFrom-Json
 if ($run061Target.name -ne $ExpectedName) { throw 'Name mismatch' }
 return $run061Target
}
$null = Read-Target
Add-Content -LiteralPath $LogPath -Value "ARMED $PodId deadline $($DeadlineUtc.ToString('o'))"
while ([DateTimeOffset]::UtcNow -lt $DeadlineUtc) {
 $run061Seconds = [Math]::Min(60, [Math]::Max(1, ($DeadlineUtc-[DateTimeOffset]::UtcNow).TotalSeconds))
 Start-Sleep -Seconds $run061Seconds
}
for ($run061Try=1; $run061Try -le 5; $run061Try++) {
 try {
  $null = Read-Target
  & $run061Cli pod stop $PodId
  if ($LASTEXITCODE -ne 0) { throw 'Stop failed' }
  $run061After = Read-Target
  if ($run061After.status -notin @('EXITED','STOPPED') -and $run061After.desiredStatus -notin @('EXITED','STOPPED')) { throw 'Stop unconfirmed' }
  Add-Content -LiteralPath $LogPath -Value "STOP_CONFIRMED $PodId; volume retained for retrieval"
  exit 0
 } catch {
  Add-Content -LiteralPath $LogPath -Value "RETRY $run061Try stop $PodId"
  Start-Sleep -Seconds 15
 }
}
exit 1

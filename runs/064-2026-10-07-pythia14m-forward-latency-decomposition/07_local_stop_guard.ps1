param(
 [Parameter(Mandatory=$true)][string]$PodId,
 [Parameter(Mandatory=$true)][string]$ExpectedName,
 [Parameter(Mandatory=$true)][DateTimeOffset]$DeadlineUtc,
 [Parameter(Mandatory=$true)][string]$RunpodctlPath,
 [Parameter(Mandatory=$true)][string]$LogPath
)
$ErrorActionPreference = 'Stop'
if ($PodId -notmatch '^[a-z0-9]+$' -or $ExpectedName -notlike 'run064-*') { throw 'Invalid scoped identity' }
$run064Cli = (Resolve-Path -LiteralPath $RunpodctlPath).Path
function Read-Target {
 $run064Raw = & $run064Cli pod get $PodId
 if ($LASTEXITCODE -ne 0) { throw 'Pod lookup failed' }
 $run064Target = $run064Raw | ConvertFrom-Json
 if ($run064Target.name -ne $ExpectedName) { throw 'Name mismatch' }
 return $run064Target
}
$null = Read-Target
Add-Content -LiteralPath $LogPath -Value "ARMED $PodId deadline $($DeadlineUtc.ToString('o'))"
while ([DateTimeOffset]::UtcNow -lt $DeadlineUtc) {
 $run064Seconds = [Math]::Min(60, [Math]::Max(1, ($DeadlineUtc-[DateTimeOffset]::UtcNow).TotalSeconds))
 Start-Sleep -Seconds $run064Seconds
}
for ($run064Try=1; $run064Try -le 5; $run064Try++) {
 try {
  $null = Read-Target
  & $run064Cli pod stop $PodId
  if ($LASTEXITCODE -ne 0) { throw 'Stop failed' }
  $run064After = Read-Target
  if ($run064After.status -notin @('EXITED','STOPPED') -and $run064After.desiredStatus -notin @('EXITED','STOPPED')) { throw 'Stop unconfirmed' }
  Add-Content -LiteralPath $LogPath -Value "STOP_CONFIRMED $PodId; volume retained for retrieval"
  exit 0
 } catch {
  Add-Content -LiteralPath $LogPath -Value "RETRY $run064Try stop $PodId"
  Start-Sleep -Seconds 15
 }
}
exit 1


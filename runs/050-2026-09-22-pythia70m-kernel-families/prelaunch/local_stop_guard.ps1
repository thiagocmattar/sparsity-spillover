param(
 [Parameter(Mandatory=$true)][string]$PodId,
 [Parameter(Mandatory=$true)][string]$ExpectedName,
 [Parameter(Mandatory=$true)][DateTimeOffset]$DeadlineUtc,
 [Parameter(Mandatory=$true)][string]$RunpodctlPath,
 [Parameter(Mandatory=$true)][string]$LogPath
)
$ErrorActionPreference = 'Stop'
if ($PodId -notmatch '^[a-z0-9]+$' -or $ExpectedName -notlike 'run049-*') { throw 'Invalid scoped identity' }
$run049Cli = (Resolve-Path -LiteralPath $RunpodctlPath).Path
function Read-Target {
 $run049Raw = & $run049Cli pod get $PodId
 if ($LASTEXITCODE -ne 0) { throw 'Pod lookup failed' }
 $run049Target = $run049Raw | ConvertFrom-Json
 if ($run049Target.name -ne $ExpectedName) { throw 'Name mismatch' }
 return $run049Target
}
$null = Read-Target
Add-Content -LiteralPath $LogPath -Value "ARMED $PodId deadline $($DeadlineUtc.ToString('o'))"
while ([DateTimeOffset]::UtcNow -lt $DeadlineUtc) {
 $run049Seconds = [Math]::Min(60, [Math]::Max(1, ($DeadlineUtc-[DateTimeOffset]::UtcNow).TotalSeconds))
 Start-Sleep -Seconds $run049Seconds
}
for ($run049Try=1; $run049Try -le 5; $run049Try++) {
 try {
  $null = Read-Target
  & $run049Cli pod stop $PodId
  if ($LASTEXITCODE -ne 0) { throw 'Stop failed' }
  $run049After = Read-Target
  if ($run049After.status -notin @('EXITED','STOPPED') -and $run049After.desiredStatus -notin @('EXITED','STOPPED')) { throw 'Stop unconfirmed' }
  Add-Content -LiteralPath $LogPath -Value "STOP_CONFIRMED $PodId; volume retained for retrieval"
  exit 0
 } catch {
  Add-Content -LiteralPath $LogPath -Value "RETRY $run049Try stop $PodId"
  Start-Sleep -Seconds 15
 }
}
exit 1

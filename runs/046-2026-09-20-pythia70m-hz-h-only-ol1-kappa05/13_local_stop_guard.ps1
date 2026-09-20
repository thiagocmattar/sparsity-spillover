param(
 [Parameter(Mandatory=$true)][string]$PodId,
 [Parameter(Mandatory=$true)][string]$ExpectedName,
 [Parameter(Mandatory=$true)][DateTimeOffset]$DeadlineUtc,
 [Parameter(Mandatory=$true)][string]$RunpodctlPath,
 [Parameter(Mandatory=$true)][string]$LogPath
)
$ErrorActionPreference = 'Stop'
if ($PodId -notmatch '^[a-z0-9]+$' -or $ExpectedName -notlike 'run046-*') { throw 'Invalid scoped identity' }
$run046Cli = (Resolve-Path -LiteralPath $RunpodctlPath).Path
function Read-Target {
 $run046Raw = & $run046Cli pod get $PodId
 if ($LASTEXITCODE -ne 0) { throw 'Pod lookup failed' }
 $run046Target = $run046Raw | ConvertFrom-Json
 if ($run046Target.name -ne $ExpectedName) { throw 'Name mismatch' }
 return $run046Target
}
$null = Read-Target
Add-Content -LiteralPath $LogPath -Value "ARMED $PodId deadline $($DeadlineUtc.ToString('o'))"
while ([DateTimeOffset]::UtcNow -lt $DeadlineUtc) {
 $run046Seconds = [Math]::Min(60, [Math]::Max(1, ($DeadlineUtc-[DateTimeOffset]::UtcNow).TotalSeconds))
 Start-Sleep -Seconds $run046Seconds
}
for ($run046Try=1; $run046Try -le 5; $run046Try++) {
 try {
  $null = Read-Target
  & $run046Cli pod stop $PodId
  if ($LASTEXITCODE -ne 0) { throw 'Stop failed' }
  $run046After = Read-Target
  if ($run046After.status -notin @('EXITED','STOPPED') -and $run046After.desiredStatus -notin @('EXITED','STOPPED')) { throw 'Stop unconfirmed' }
  Add-Content -LiteralPath $LogPath -Value "STOP_CONFIRMED $PodId; volume retained for retrieval"
  exit 0
 } catch {
  Add-Content -LiteralPath $LogPath -Value "RETRY $run046Try stop $PodId"
  Start-Sleep -Seconds 15
 }
}
exit 1

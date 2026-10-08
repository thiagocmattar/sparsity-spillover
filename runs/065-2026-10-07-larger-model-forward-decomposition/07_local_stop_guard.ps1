param(
 [Parameter(Mandatory=$true)][string]$PodId,
 [Parameter(Mandatory=$true)][string]$ExpectedName,
 [Parameter(Mandatory=$true)][DateTimeOffset]$DeadlineUtc,
 [Parameter(Mandatory=$true)][string]$RunpodctlPath,
 [Parameter(Mandatory=$true)][string]$LogPath
)
$ErrorActionPreference = 'Stop'
if ($PodId -notmatch '^[a-z0-9]+$' -or $ExpectedName -notlike 'run065-*') { throw 'Invalid scoped identity' }
$run065Cli = (Resolve-Path -LiteralPath $RunpodctlPath).Path
function Read-Target {
 $run065Raw = & $run065Cli pod get $PodId
 if ($LASTEXITCODE -ne 0) { throw 'Pod lookup failed' }
 $run065Target = $run065Raw | ConvertFrom-Json
 if ($run065Target.name -ne $ExpectedName) { throw 'Name mismatch' }
 return $run065Target
}
$null = Read-Target
Add-Content -LiteralPath $LogPath -Value "ARMED $PodId deadline $($DeadlineUtc.ToString('o'))"
while ([DateTimeOffset]::UtcNow -lt $DeadlineUtc) {
 $run065Seconds = [Math]::Min(60, [Math]::Max(1, ($DeadlineUtc-[DateTimeOffset]::UtcNow).TotalSeconds))
 Start-Sleep -Seconds $run065Seconds
}
for ($run065Try=1; $run065Try -le 5; $run065Try++) {
 try {
  $null = Read-Target
  & $run065Cli pod stop $PodId
  if ($LASTEXITCODE -ne 0) { throw 'Stop failed' }
  $run065After = Read-Target
  if ($run065After.status -notin @('EXITED','STOPPED') -and $run065After.desiredStatus -notin @('EXITED','STOPPED')) { throw 'Stop unconfirmed' }
  Add-Content -LiteralPath $LogPath -Value "STOP_CONFIRMED $PodId; volume retained for retrieval"
  exit 0
 } catch {
  Add-Content -LiteralPath $LogPath -Value "RETRY $run065Try stop $PodId"
  Start-Sleep -Seconds 15
 }
}
exit 1


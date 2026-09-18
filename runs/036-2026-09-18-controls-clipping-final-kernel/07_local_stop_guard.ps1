param(
 [Parameter(Mandatory=$true)][string]$PodId,
 [Parameter(Mandatory=$true)][string]$ExpectedName,
 [Parameter(Mandatory=$true)][DateTimeOffset]$DeadlineUtc,
 [Parameter(Mandatory=$true)][string]$RunpodctlPath,
 [Parameter(Mandatory=$true)][string]$LogPath
)
$ErrorActionPreference = 'Stop'
if ($PodId -notmatch '^[a-z0-9]+$' -or $ExpectedName -notlike 'run036-*') { throw 'Invalid scoped identity' }
$run036Cli = (Resolve-Path -LiteralPath $RunpodctlPath).Path
function Read-Target {
 $run036Raw = & $run036Cli pod get $PodId
 if ($LASTEXITCODE -ne 0) { throw 'Pod lookup failed' }
 $run036Target = $run036Raw | ConvertFrom-Json
 if ($run036Target.name -ne $ExpectedName) { throw 'Name mismatch' }
 return $run036Target
}
$null = Read-Target
Add-Content -LiteralPath $LogPath -Value "ARMED $PodId deadline $($DeadlineUtc.ToString('o'))"
while ([DateTimeOffset]::UtcNow -lt $DeadlineUtc) {
 $run036Remaining = ($DeadlineUtc - [DateTimeOffset]::UtcNow).TotalSeconds
 Start-Sleep -Seconds ([Math]::Min(30, [Math]::Max(1, [Math]::Ceiling($run036Remaining))))
}
for ($run036Try=1; $run036Try -le 5; $run036Try++) {
 try {
  $null = Read-Target
  & $run036Cli pod stop $PodId
  if ($LASTEXITCODE -ne 0) { throw 'Stop failed' }
  $run036After = Read-Target
  if (($run036After.desiredStatus -notin @('EXITED','STOPPED')) -and ($run036After.status -notin @('EXITED','STOPPED'))) { throw 'Stop unconfirmed' }
  Add-Content -LiteralPath $LogPath -Value "STOP_CONFIRMED $PodId; volume retained for retrieval"
  exit 0
 } catch {
  Add-Content -LiteralPath $LogPath -Value "RETRY $run036Try stop $PodId"
  Start-Sleep -Seconds 15
 }
}
exit 1

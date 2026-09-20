param(
 [Parameter(Mandatory=$true)][string]$PodId,
 [Parameter(Mandatory=$true)][string]$ExpectedName,
 [Parameter(Mandatory=$true)][DateTimeOffset]$DeadlineUtc,
 [Parameter(Mandatory=$true)][string]$RunpodctlPath,
 [Parameter(Mandatory=$true)][string]$LogPath
)
$ErrorActionPreference = 'Stop'
if ($PodId -notmatch '^[a-z0-9]+$' -or $ExpectedName -notlike 'run045-*') { throw 'Invalid scoped identity' }
$run045Cli = (Resolve-Path -LiteralPath $RunpodctlPath).Path
function Read-Target {
 $run045Raw = & $run045Cli pod get $PodId
 if ($LASTEXITCODE -ne 0) { throw 'Pod lookup failed' }
 $run045Target = $run045Raw | ConvertFrom-Json
 if ($run045Target.name -ne $ExpectedName) { throw 'Name mismatch' }
 return $run045Target
}
$null = Read-Target
Add-Content -LiteralPath $LogPath -Value "ARMED $PodId deadline $($DeadlineUtc.ToString('o'))"
while ([DateTimeOffset]::UtcNow -lt $DeadlineUtc) {
 $run045Seconds = [Math]::Min(60, [Math]::Max(1, ($DeadlineUtc-[DateTimeOffset]::UtcNow).TotalSeconds))
 Start-Sleep -Seconds $run045Seconds
}
for ($run045Try=1; $run045Try -le 5; $run045Try++) {
 try {
  $null = Read-Target
  & $run045Cli pod stop $PodId
  if ($LASTEXITCODE -ne 0) { throw 'Stop failed' }
  $run045After = Read-Target
  if ($run045After.status -notin @('EXITED','STOPPED') -and $run045After.desiredStatus -notin @('EXITED','STOPPED')) { throw 'Stop unconfirmed' }
  Add-Content -LiteralPath $LogPath -Value "STOP_CONFIRMED $PodId; volume retained for retrieval"
  exit 0
 } catch {
  Add-Content -LiteralPath $LogPath -Value "RETRY $run045Try stop $PodId"
  Start-Sleep -Seconds 15
 }
}
exit 1

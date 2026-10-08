param(
 [Parameter(Mandatory=$true)][string]$PodId,
 [Parameter(Mandatory=$true)][string]$ExpectedName,
 [Parameter(Mandatory=$true)][DateTimeOffset]$DeadlineUtc,
 [Parameter(Mandatory=$true)][string]$RunpodctlPath,
 [Parameter(Mandatory=$true)][string]$LogPath
)
$ErrorActionPreference = 'Stop'
if ($PodId -notmatch '^[a-z0-9]+$' -or $ExpectedName -notlike 'run068-*') { throw 'Invalid scoped identity' }
$run068Cli = (Resolve-Path -LiteralPath $RunpodctlPath).Path
function Read-Target {
 $run068Raw = & $run068Cli pod get $PodId
 if ($LASTEXITCODE -ne 0) { throw 'Pod lookup failed' }
 $run068Target = $run068Raw | ConvertFrom-Json
 if ($run068Target.name -ne $ExpectedName) { throw 'Name mismatch' }
 return $run068Target
}
$null = Read-Target
Add-Content -LiteralPath $LogPath -Value "ARMED $PodId deadline $($DeadlineUtc.ToString('o'))"
while ([DateTimeOffset]::UtcNow -lt $DeadlineUtc) {
 $run068Seconds = [Math]::Min(60, [Math]::Max(1, ($DeadlineUtc-[DateTimeOffset]::UtcNow).TotalSeconds))
 Start-Sleep -Seconds $run068Seconds
}
for ($run068Try=1; $run068Try -le 5; $run068Try++) {
 try {
  $null = Read-Target
  & $run068Cli pod stop $PodId
  if ($LASTEXITCODE -ne 0) { throw 'Stop failed' }
  $run068After = Read-Target
  if ($run068After.status -notin @('EXITED','STOPPED') -and $run068After.desiredStatus -notin @('EXITED','STOPPED')) { throw 'Stop unconfirmed' }
  Add-Content -LiteralPath $LogPath -Value "STOP_CONFIRMED $PodId; volume retained for retrieval"
  exit 0
 } catch {
  Add-Content -LiteralPath $LogPath -Value "RETRY $run068Try stop $PodId"
  Start-Sleep -Seconds 15
 }
}
exit 1

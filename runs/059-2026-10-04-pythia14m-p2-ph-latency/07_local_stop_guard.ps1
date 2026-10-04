param(
 [Parameter(Mandatory=$true)][string]$PodId,
 [Parameter(Mandatory=$true)][string]$ExpectedName,
 [Parameter(Mandatory=$true)][DateTimeOffset]$DeadlineUtc,
 [Parameter(Mandatory=$true)][string]$RunpodctlPath,
 [Parameter(Mandatory=$true)][string]$LogPath
)
$ErrorActionPreference = 'Stop'
if ($PodId -notmatch '^[a-z0-9]+$' -or $ExpectedName -notlike 'run059-*') { throw 'Invalid scoped identity' }
$run059Cli = (Resolve-Path -LiteralPath $RunpodctlPath).Path
function Read-Target {
 $run059Raw = & $run059Cli pod get $PodId
 if ($LASTEXITCODE -ne 0) { throw 'Pod lookup failed' }
 $run059Target = $run059Raw | ConvertFrom-Json
 if ($run059Target.name -ne $ExpectedName) { throw 'Name mismatch' }
 return $run059Target
}
$null = Read-Target
Add-Content -LiteralPath $LogPath -Value "ARMED $PodId deadline $($DeadlineUtc.ToString('o'))"
while ([DateTimeOffset]::UtcNow -lt $DeadlineUtc) {
 $run059Seconds = [Math]::Min(60, [Math]::Max(1, ($DeadlineUtc-[DateTimeOffset]::UtcNow).TotalSeconds))
 Start-Sleep -Seconds $run059Seconds
}
for ($run059Try=1; $run059Try -le 5; $run059Try++) {
 try {
  $null = Read-Target
  & $run059Cli pod stop $PodId
  if ($LASTEXITCODE -ne 0) { throw 'Stop failed' }
  $run059After = Read-Target
  if ($run059After.status -notin @('EXITED','STOPPED') -and $run059After.desiredStatus -notin @('EXITED','STOPPED')) { throw 'Stop unconfirmed' }
  Add-Content -LiteralPath $LogPath -Value "STOP_CONFIRMED $PodId; volume retained for retrieval"
  exit 0
 } catch {
  Add-Content -LiteralPath $LogPath -Value "RETRY $run059Try stop $PodId"
  Start-Sleep -Seconds 15
 }
}
exit 1

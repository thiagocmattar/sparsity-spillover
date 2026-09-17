param(
 [Parameter(Mandatory=$true)][int]$ControllerPid,
 [Parameter(Mandatory=$true)][DateTimeOffset]$DeadlineUtc
)
$ErrorActionPreference='Stop'
$run034Controller=Get-CimInstance Win32_Process -Filter "ProcessId = $ControllerPid"
if ($run034Controller.CommandLine -notlike '*034-2026-09-17-pythia70m-h-only-ol1*monitor_and_retrieve.py*') {
 throw 'Run034 recovery-controller identity mismatch'
}
Add-Type -TypeDefinition @'
using System.Runtime.InteropServices;
public static class Run034PowerRequest {
 [DllImport("kernel32.dll")]
 public static extern uint SetThreadExecutionState(uint flags);
}
'@
# Process-scoped system-awake request; no permanent power-setting or display change.
if ([Run034PowerRequest]::SetThreadExecutionState([uint32]2147483649) -eq 0) {
 throw 'Could not hold the system awake for checkpoint recovery'
}
try {
 while ([DateTimeOffset]::UtcNow -lt $DeadlineUtc -and (Get-Process -Id $ControllerPid -ErrorAction SilentlyContinue)) {
  Start-Sleep -Seconds 30
 }
} finally {
 $null=[Run034PowerRequest]::SetThreadExecutionState([uint32]2147483648)
}

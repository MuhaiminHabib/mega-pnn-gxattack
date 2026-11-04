$ErrorActionPreference = 'Stop'
$taskRoot = $PSScriptRoot
$taskPython = 'C:\habib dissertation\Projects\AML-GNN-Replication-v2\.venv\Scripts\python.exe'
$taskExisting = @(Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^python' -and $_.CommandLine -match 'mega_retraining_pilot_v1[\\/]supervise\.py' })
if ($taskExisting.Count -gt 0) { throw 'Retraining controller already running.' }
if (Test-Path -LiteralPath (Join-Path $taskRoot 'independent_verification.json')) { throw 'Already complete.' }
& $taskPython -B -u (Join-Path $taskRoot 'supervise.py') >> (Join-Path $taskRoot 'controller.log') 2>> (Join-Path $taskRoot 'controller_errors.log')
exit $LASTEXITCODE

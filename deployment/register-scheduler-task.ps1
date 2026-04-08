<#
.SYNOPSIS
    Registers a Windows Task Scheduler job that auto-starts the
    Centurion Core scheduler every weekday at 9:00 AM IST (03:30 UTC).

.DESCRIPTION
    - Runs under the current user account
    - Activates the Python venv, then launches scheduler.py
    - Restarts on failure (up to 3 retries, 60s delay)
    - Logs stdout/stderr to data/scheduler_task.log
    - To remove: Unregister-ScheduledTask -TaskName "CenturionScheduler"

.NOTES
    Run this script ONCE (elevated / admin) to register the task.
    The scheduler itself runs the full cron-based pipeline internally.
#>

$ErrorActionPreference = "Stop"

$TaskName  = "CenturionScheduler"
$VenvDir   = "c:\Users\suraboyi\Videos\dev_algo\myenv"
$CoreDir   = "c:\Users\suraboyi\Videos\dev_algo\centurion_core"
$LogFile   = Join-Path $CoreDir "data\scheduler_task.log"

# Ensure log dir exists
$logDir = Split-Path $LogFile -Parent
if (-not (Test-Path $logDir)) { New-Item -ItemType Directory -Path $logDir -Force | Out-Null }

# Build the command: activate venv -> run scheduler -> pipe logs
$PythonExe = Join-Path $VenvDir "Scripts\python.exe"
$ScriptPath = Join-Path $CoreDir "scheduler.py"

# Verify paths
if (-not (Test-Path $PythonExe)) { throw "Python not found at $PythonExe" }
if (-not (Test-Path $ScriptPath)) { throw "scheduler.py not found at $ScriptPath" }

# FIX: Task Scheduler doesn't interpret shell redirects (>>, 2>&1).
# Wrap in cmd.exe /c so the redirect syntax is processed by a shell.
$CmdLine = "`"$PythonExe`" -u `"$ScriptPath`" >> `"$LogFile`" 2>&1"

Write-Host "Registering Windows Task: $TaskName" -ForegroundColor Cyan
Write-Host "  Python : $PythonExe"
Write-Host "  Script : $ScriptPath"
Write-Host "  Log    : $LogFile"

# Remove existing task if present
$existing = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
if ($existing) {
    Write-Host "  Removing existing task..." -ForegroundColor Yellow
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
}

# -- Trigger: Mon-Fri at 08:45 AM (IST) - 15 min before market scan --
$Trigger = New-ScheduledTaskTrigger -Weekly `
    -DaysOfWeek Monday, Tuesday, Wednesday, Thursday, Friday `
    -At "08:45"

# -- Action: run python scheduler.py via cmd.exe for proper redirect handling --
$Action = New-ScheduledTaskAction `
    -Execute "cmd.exe" `
    -Argument "/c $CmdLine" `
    -WorkingDirectory $CoreDir

# -- Settings: restart on failure, don't stop if on battery --
$Settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -RestartCount 3 `
    -RestartInterval (New-TimeSpan -Minutes 1) `
    -ExecutionTimeLimit (New-TimeSpan -Hours 10)

# -- Register --
Register-ScheduledTask `
    -TaskName $TaskName `
    -Trigger $Trigger `
    -Action $Action `
    -Settings $Settings `
    -Description "Centurion Core - APScheduler pipeline (Mon-Fri 08:45 IST)" `
    -RunLevel Highest

Write-Host ""
Write-Host "Task '$TaskName' registered." -ForegroundColor Green
Write-Host ""
Write-Host "Verify with:" -ForegroundColor Cyan
Write-Host "  Get-ScheduledTask -TaskName '$TaskName' | Format-List"
Write-Host ""
Write-Host "To start immediately:" -ForegroundColor Cyan
Write-Host "  Start-ScheduledTask -TaskName '$TaskName'"
Write-Host ""
Write-Host "To remove:" -ForegroundColor Cyan
Write-Host '  Unregister-ScheduledTask -TaskName "CenturionScheduler" -Confirm:$false'

# Save this file to D:\Cursor\install.ps1 and run once to create all scripts locally.
# Run: powershell -NoProfile -ExecutionPolicy Bypass -File D:\Cursor\install.ps1

$dest = 'D:\Cursor'
New-Item -ItemType Directory -Path $dest -Force | Out-Null

$migratePs1 = @'
#Requires -Version 5.1
$ErrorActionPreference = 'Stop'
$TargetRoot = 'D:\Cursor\UserData'
$LogFile = Join-Path $TargetRoot 'migration.log'
function Write-Log { param([string]$Message)
  $line = "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] $Message"
  Write-Host $line
  $logDir = Split-Path $LogFile -Parent
  if (Test-Path -LiteralPath $logDir) { Add-Content -Path $LogFile -Value $line -Encoding UTF8 }
}
function Test-IsJunction { param([string]$Path)
  if (-not (Test-Path -LiteralPath $Path)) { return $false }
  return ((Get-Item -LiteralPath $Path -Force).Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0
}
function Stop-CursorProcesses {
  foreach ($name in @('Cursor','cursor')) {
    Get-Process -Name $name -ErrorAction SilentlyContinue | Stop-Process -Force
  }
  Start-Sleep -Seconds 3
  foreach ($name in @('Cursor','cursor')) {
    if (Get-Process -Name $name -ErrorAction SilentlyContinue) {
      throw 'Cursor is still running. Close it in Task Manager, then retry.'
    }
  }
}
function Migrate-Folder { param([string]$Source,[string]$Dest)
  if (-not (Test-Path -LiteralPath $Source)) { Write-Log "SKIP: $Source"; return }
  if (Test-IsJunction -Path $Source) { Write-Log "SKIP junction: $Source"; return }
  if (Test-Path -LiteralPath $Dest) { throw "Target exists: $Dest" }
  $p = Split-Path $Dest -Parent
  if (-not (Test-Path -LiteralPath $p)) { New-Item -ItemType Directory -Path $p -Force | Out-Null }
  Write-Log "MOVE: $Source -> $Dest"
  Move-Item -LiteralPath $Source -Destination $Dest
  Write-Log "LINK: $Source -> $Dest"
  cmd /c mklink /J "$Source" "$Dest" | Out-Null
  if ($LASTEXITCODE -ne 0) { throw "mklink failed: $Source" }
}
New-Item -ItemType Directory -Path $TargetRoot -Force | Out-Null
Write-Log 'START'
Stop-CursorProcesses
Migrate-Folder -Source "$env:APPDATA\Cursor" -Dest "$TargetRoot\AppData\Roaming"
Migrate-Folder -Source "$env:LOCALAPPDATA\Cursor" -Dest "$TargetRoot\AppData\Local"
Migrate-Folder -Source "$env:USERPROFILE\.cursor" -Dest "$TargetRoot\UserProfile\.cursor"
Write-Log 'DONE'
Write-Host 'Migration complete. Restart Cursor.' -ForegroundColor Green
'@

$rollbackPs1 = @'
#Requires -Version 5.1
$ErrorActionPreference = 'Stop'
$TargetRoot = 'D:\Cursor\UserData'
function Restore-Folder { param([string]$Link,[string]$Backup)
  if (-not (Test-Path -LiteralPath $Link)) { return }
  $item = Get-Item -LiteralPath $Link -Force
  if (($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -eq 0) { return }
  if (-not (Test-Path -LiteralPath $Backup)) { throw "Backup missing: $Backup" }
  Remove-Item -LiteralPath $Link -Force
  Move-Item -LiteralPath $Backup -Destination $Link
  Write-Host "RESTORED: $Link" -ForegroundColor Green
}
foreach ($name in @('Cursor','cursor')) {
  Get-Process -Name $name -ErrorAction SilentlyContinue | Stop-Process -Force
}
Start-Sleep -Seconds 2
Restore-Folder -Link "$env:APPDATA\Cursor" -Backup "$TargetRoot\AppData\Roaming"
Restore-Folder -Link "$env:LOCALAPPDATA\Cursor" -Backup "$TargetRoot\AppData\Local"
Restore-Folder -Link "$env:USERPROFILE\.cursor" -Backup "$TargetRoot\UserProfile\.cursor"
Write-Host 'Rollback complete.' -ForegroundColor Green
'@

$migrateBat = @'
@echo off
cd /d "%~dp0"
net session >nul 2>&1
if %errorLevel% neq 0 (
    powershell -NoProfile -Command "Start-Process cmd -ArgumentList '/c \"\"%~f0\"\"' -Verb RunAs"
    exit /b
)
echo Quit Cursor first, then press any key...
pause >nul
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0migrate-cursor.ps1"
pause
'@

$rollbackBat = @'
@echo off
cd /d "%~dp0"
net session >nul 2>&1
if %errorLevel% neq 0 (
    powershell -NoProfile -Command "Start-Process cmd -ArgumentList '/c \"\"%~f0\"\"' -Verb RunAs"
    exit /b
)
echo Quit Cursor first, then press any key...
pause >nul
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0rollback-cursor.ps1"
pause
'@

Set-Content -Path (Join-Path $dest 'migrate-cursor.ps1')  -Value $migratePs1  -Encoding ASCII
Set-Content -Path (Join-Path $dest 'rollback-cursor.ps1') -Value $rollbackPs1 -Encoding ASCII
Set-Content -Path (Join-Path $dest 'migrate-cursor.bat')  -Value $migrateBat  -Encoding ASCII
Set-Content -Path (Join-Path $dest 'rollback-cursor.bat') -Value $rollbackBat -Encoding ASCII

Write-Host "Scripts written to $dest" -ForegroundColor Green
Write-Host "Run: D:\Cursor\migrate-cursor.bat (as admin)"

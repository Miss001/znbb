#Requires -Version 5.1
# Rollback Cursor user data migration
# Quit Cursor completely before running. Admin required.

$ErrorActionPreference = 'Stop'

$TargetRoot = 'D:\Cursor\UserData'

function Restore-Folder {
    param(
        [string]$Link,
        [string]$Backup
    )

    if (-not (Test-Path -LiteralPath $Link)) {
        Write-Host "SKIP (missing): $Link"
        return
    }

    $item = Get-Item -LiteralPath $Link -Force
    if (($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -eq 0) {
        Write-Host "SKIP (not a junction): $Link"
        return
    }

    if (-not (Test-Path -LiteralPath $Backup)) {
        throw "Backup not found: $Backup"
    }

    Remove-Item -LiteralPath $Link -Force
    Move-Item -LiteralPath $Backup -Destination $Link
    Write-Host "RESTORED: $Link" -ForegroundColor Green
}

$names = @('Cursor', 'cursor')
foreach ($name in $names) {
    $procs = Get-Process -Name $name -ErrorAction SilentlyContinue
    if ($procs) {
        $procs | Stop-Process -Force
    }
}
Start-Sleep -Seconds 2

Write-Host '========== ROLLBACK START ==========' -ForegroundColor Yellow

Restore-Folder -Link "$env:APPDATA\Cursor"      -Backup "$TargetRoot\AppData\Roaming"
Restore-Folder -Link "$env:LOCALAPPDATA\Cursor" -Backup "$TargetRoot\AppData\Local"
Restore-Folder -Link "$env:USERPROFILE\.cursor" -Backup "$TargetRoot\UserProfile\.cursor"

Write-Host ''
Write-Host 'Rollback complete. Restart Cursor.' -ForegroundColor Green

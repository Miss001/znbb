#Requires -Version 5.1
# Cursor user data migration -> D:\Cursor\UserData
# Quit Cursor completely before running. Admin required.

$ErrorActionPreference = 'Stop'

$TargetRoot = 'D:\Cursor\UserData'
$LogFile    = Join-Path $TargetRoot 'migration.log'

function Write-Log {
    param([string]$Message)
    $line = "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] $Message"
    Write-Host $line
    $logDir = Split-Path $LogFile -Parent
    if (Test-Path -LiteralPath $logDir) {
        Add-Content -Path $LogFile -Value $line -Encoding UTF8
    }
}

function Test-IsJunction {
    param([string]$Path)
    if (-not (Test-Path -LiteralPath $Path)) { return $false }
    $item = Get-Item -LiteralPath $Path -Force
    return ($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0
}

function Stop-CursorProcesses {
    $names = @('Cursor', 'cursor')
    foreach ($name in $names) {
        $procs = Get-Process -Name $name -ErrorAction SilentlyContinue
        if ($procs) {
            $procs | Stop-Process -Force
        }
    }
    Start-Sleep -Seconds 3
    foreach ($name in $names) {
        $running = Get-Process -Name $name -ErrorAction SilentlyContinue
        if ($running) {
            throw 'Cursor is still running. Close all Cursor processes in Task Manager, then retry.'
        }
    }
}

function Migrate-Folder {
    param(
        [string]$Source,
        [string]$Dest
    )

    if (-not (Test-Path -LiteralPath $Source)) {
        Write-Log "SKIP (source missing): $Source"
        return
    }

    if (Test-IsJunction -Path $Source) {
        Write-Log "SKIP (already junction): $Source"
        return
    }

    if (Test-Path -LiteralPath $Dest) {
        throw "Target already exists: $Dest"
    }

    $destParent = Split-Path $Dest -Parent
    if (-not (Test-Path -LiteralPath $destParent)) {
        New-Item -ItemType Directory -Path $destParent -Force | Out-Null
    }

    Write-Log "MOVE: $Source -> $Dest"
    Move-Item -LiteralPath $Source -Destination $Dest

    Write-Log "LINK: $Source -> $Dest"
    cmd /c mklink /J "$Source" "$Dest" | Out-Null
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to create junction: $Source -> $Dest"
    }
}

New-Item -ItemType Directory -Path $TargetRoot -Force | Out-Null
Write-Log '========== START =========='

Stop-CursorProcesses

Migrate-Folder -Source "$env:APPDATA\Cursor"      -Dest "$TargetRoot\AppData\Roaming"
Migrate-Folder -Source "$env:LOCALAPPDATA\Cursor" -Dest "$TargetRoot\AppData\Local"
Migrate-Folder -Source "$env:USERPROFILE\.cursor" -Dest "$TargetRoot\UserProfile\.cursor"

Write-Log '========== DONE =========='
Write-Log "Data dir: $TargetRoot"
Write-Host ''
Write-Host 'Migration complete. Restart Cursor.' -ForegroundColor Green
Write-Host "Log: $LogFile"

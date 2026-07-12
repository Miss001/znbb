#Requires -Version 5.1
<#
.SYNOPSIS
    将 Cursor 用户数据迁移到 D:\Cursor\UserData，并通过目录联接保持原路径可用。

.DESCRIPTION
    迁移范围：
      %APPDATA%\Cursor          -> D:\Cursor\UserData\AppData\Roaming
      %LOCALAPPDATA%\Cursor     -> D:\Cursor\UserData\AppData\Local
      %USERPROFILE%\.cursor       -> D:\Cursor\UserData\UserProfile\.cursor

    使用前请先完全退出 Cursor（含任务管理器中的残留进程）。
    需要管理员权限（创建目录联接）。

.NOTES
    编码: UTF-8
    日志: D:\Cursor\UserData\migration.log
#>

$ErrorActionPreference = 'Stop'

$TargetRoot = 'D:\Cursor\UserData'
$LogFile    = Join-Path $TargetRoot 'migration.log'

function Write-Log {
    param([string]$Message)
    $line = "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] $Message"
    Write-Host $line
    if (Test-Path -LiteralPath (Split-Path $LogFile -Parent)) {
        Add-Content -Path $LogFile -Value $line -Encoding UTF8
    }
}

function Test-IsJunction {
    param([string]$Path)
    if (-not (Test-Path -LiteralPath $Path)) { return $false }
    return ((Get-Item -LiteralPath $Path -Force).Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0
}

function Stop-CursorProcesses {
    $names = @('Cursor', 'cursor')
    foreach ($name in $names) {
        Get-Process -Name $name -ErrorAction SilentlyContinue | Stop-Process -Force
    }
    Start-Sleep -Seconds 3
    foreach ($name in $names) {
        if (Get-Process -Name $name -ErrorAction SilentlyContinue) {
            throw '仍有 Cursor 进程在运行，请先在任务管理器中结束所有 Cursor 相关进程后重试。'
        }
    }
}

function Migrate-Folder {
    param(
        [string]$Source,
        [string]$Dest
    )

    if (-not (Test-Path -LiteralPath $Source)) {
        Write-Log "跳过（源不存在）: $Source"
        return
    }

    if (Test-IsJunction $Source) {
        Write-Log "跳过（已是联接）: $Source"
        return
    }

    if (Test-Path -LiteralPath $Dest) {
        throw "目标已存在，请先检查后再执行: $Dest"
    }

    $destParent = Split-Path $Dest -Parent
    if (-not (Test-Path -LiteralPath $destParent)) {
        New-Item -ItemType Directory -Path $destParent -Force | Out-Null
    }

    Write-Log "移动: $Source -> $Dest"
    Move-Item -LiteralPath $Source -Destination $Dest

    Write-Log "创建联接: $Source -> $Dest"
    $null = cmd /c mklink /J "$Source" "$Dest"
    if ($LASTEXITCODE -ne 0) {
        throw "创建联接失败: $Source -> $Dest"
    }
}

# ---- 主流程 ----

New-Item -ItemType Directory -Path $TargetRoot -Force | Out-Null
Write-Log '========== 开始迁移 =========='

Stop-CursorProcesses

Migrate-Folder -Source "$env:APPDATA\Cursor"        -Dest "$TargetRoot\AppData\Roaming"
Migrate-Folder -Source "$env:LOCALAPPDATA\Cursor"   -Dest "$TargetRoot\AppData\Local"
Migrate-Folder -Source "$env:USERPROFILE\.cursor"   -Dest "$TargetRoot\UserProfile\.cursor"

Write-Log '========== 迁移完成 =========='
Write-Log "数据目录: $TargetRoot"
Write-Host ''
Write-Host '迁移成功！请重新启动 Cursor。' -ForegroundColor Green
Write-Host "日志文件: $LogFile"

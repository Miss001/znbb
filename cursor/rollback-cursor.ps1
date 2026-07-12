#Requires -Version 5.1
<#
.SYNOPSIS
    回滚 Cursor 用户数据迁移，将联接还原为本地目录。

.DESCRIPTION
    将 D:\Cursor\UserData 中的数据移回原始路径，并删除目录联接。
    使用前请先完全退出 Cursor。

.NOTES
    编码: UTF-8
#>

$ErrorActionPreference = 'Stop'

$TargetRoot = 'D:\Cursor\UserData'

function Restore-Folder {
    param(
        [string]$Link,
        [string]$Backup
    )

    if (-not (Test-Path -LiteralPath $Link)) {
        Write-Host "跳过（路径不存在）: $Link"
        return
    }

    $item = Get-Item -LiteralPath $Link -Force
    if (($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -eq 0) {
        Write-Host "跳过（非联接，无需回滚）: $Link"
        return
    }

    if (-not (Test-Path -LiteralPath $Backup)) {
        throw "备份数据不存在: $Backup"
    }

    Remove-Item -LiteralPath $Link -Force
    Move-Item -LiteralPath $Backup -Destination $Link
    Write-Host "已恢复: $Link" -ForegroundColor Green
}

# ---- 主流程 ----

Get-Process -Name Cursor, cursor -ErrorAction SilentlyContinue | Stop-Process -Force
Start-Sleep -Seconds 2

Write-Host '========== 开始回滚 ==========' -ForegroundColor Yellow

Restore-Folder -Link "$env:APPDATA\Cursor"      -Backup "$TargetRoot\AppData\Roaming"
Restore-Folder -Link "$env:LOCALAPPDATA\Cursor" -Backup "$TargetRoot\AppData\Local"
Restore-Folder -Link "$env:USERPROFILE\.cursor" -Backup "$TargetRoot\UserProfile\.cursor"

Write-Host ''
Write-Host '回滚完成！请重新启动 Cursor。' -ForegroundColor Green

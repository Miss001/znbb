@echo off
chcp 65001 >nul
echo ========================================
echo   Cursor 用户数据迁移工具
echo   目标: D:\Cursor\UserData
echo ========================================
echo.
echo 请确保已完全退出 Cursor，然后按任意键继续...
pause >nul

powershell -NoProfile -ExecutionPolicy Bypass -Command "Start-Process powershell -ArgumentList '-NoProfile -ExecutionPolicy Bypass -File \"%~dp0migrate-cursor.ps1\"' -Verb RunAs -Wait"

echo.
pause

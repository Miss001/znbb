@echo off
cd /d "%~dp0"
echo Downloading latest scripts to D:\Cursor ...
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0update-scripts.ps1"
echo.
pause

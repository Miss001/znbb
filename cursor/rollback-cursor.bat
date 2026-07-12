@echo off
cd /d "%~dp0"
net session >nul 2>&1
if %errorLevel% neq 0 (
    powershell -NoProfile -Command "Start-Process cmd -ArgumentList '/c \"\"%~f0\"\"' -Verb RunAs"
    exit /b
)
echo ========================================
echo   Cursor User Data Migration Rollback
echo ========================================
echo.
echo Please quit Cursor completely before continuing.
echo Press any key to start...
pause >nul
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0rollback-cursor.ps1"
echo.
pause

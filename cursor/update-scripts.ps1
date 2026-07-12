#Requires -Version 5.1
$ErrorActionPreference = 'Stop'

$dest = 'D:\Cursor'
$base = 'https://raw.githubusercontent.com/Miss001/znbb/cursor/cursor-userdata-migration-script-6fba/cursor'
$files = @(
    'migrate-cursor.ps1',
    'rollback-cursor.ps1',
    'migrate-cursor.bat',
    'rollback-cursor.bat',
    'update-scripts.ps1',
    'update-scripts.bat'
)

New-Item -ItemType Directory -Path $dest -Force | Out-Null

foreach ($file in $files) {
    $url = "$base/$file"
    $out = Join-Path $dest $file
    Write-Host "Downloading $file ..."
    Invoke-WebRequest -Uri $url -OutFile $out -UseBasicParsing
}

Write-Host ''
Write-Host "Done. Scripts saved to $dest" -ForegroundColor Green

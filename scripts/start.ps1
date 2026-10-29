# Start X-Sample: build the UI if needed, run the engine, open the browser.
# Usage: powershell -ExecutionPolicy Bypass -File scripts\start.ps1 [-Port 8765] [-NoBrowser]
param(
    [int]$Port = 8765,
    [switch]$NoBrowser
)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot

if (-not (Test-Path "$root\ui\dist\index.html")) {
    Write-Host "Building UI..."
    Push-Location "$root\ui"
    try {
        npm run build
        if ($LASTEXITCODE -ne 0) { throw "UI build failed" }
    } finally {
        Pop-Location
    }
}

$serveArgs = @("run", "--project", "$root\engine", "xsample", "serve", "--port", $Port)
if (-not $NoBrowser) { $serveArgs += "--open" }
uv @serveArgs

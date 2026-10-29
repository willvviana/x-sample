# Development mode: engine + Vite dev server (hot reload) sharing one launch token.
# Usage: powershell -ExecutionPolicy Bypass -File scripts\dev.ps1 [-Port 8765]
param([int]$Port = 8765)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot

$bytes = New-Object byte[] 32
[Security.Cryptography.RandomNumberGenerator]::Create().GetBytes($bytes)
$token = [Convert]::ToBase64String($bytes).TrimEnd("=").Replace("+", "-").Replace("/", "_")

$env:XSAMPLE_TOKEN = $token
$engine = Start-Process -PassThru -NoNewWindow -FilePath "uv" `
    -ArgumentList @("run", "--project", "`"$root\engine`"", "xsample", "serve", "--port", $Port)
Remove-Item Env:XSAMPLE_TOKEN

try {
    Start-Process "http://127.0.0.1:5173/#token=$token"
    Push-Location "$root\ui"
    $env:VITE_ENGINE_URL = "http://127.0.0.1:$Port"
    npm run dev
} finally {
    Pop-Location
    # Kill the engine and its worker processes.
    taskkill /T /F /PID $engine.Id | Out-Null
}

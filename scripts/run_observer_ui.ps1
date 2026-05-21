param(
    [string]$HostName = "127.0.0.1",
    [int]$Port = 8765,
    [switch]$Open
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $Python)) {
    $Python = "python"
}

$Url = "http://${HostName}:${Port}"
Write-Host "[observer_ui] starting read-only dashboard at $Url"
Write-Host "[observer_ui] repo root: $RepoRoot"

if ($Open) {
    Start-Process -FilePath $Url | Out-Null
}

& $Python (Join-Path $RepoRoot "observer_ui\server.py") --host $HostName --port $Port

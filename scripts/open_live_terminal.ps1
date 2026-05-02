$ErrorActionPreference = "Stop"
$projectRoot = "C:\3d-recon-pipeline"
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
$script = Join-Path $projectRoot "scripts\watch_live.py"
$pwsh7 = "C:\Program Files\PowerShell\7\pwsh.exe"
$powershell = if (Test-Path $pwsh7) { $pwsh7 } else { "C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe" }

if (!(Test-Path $python)) {
  throw "找不到 Python：$python"
}

if (!(Test-Path $script)) {
  throw "找不到 watcher：$script"
}

$bootstrap = @"
`$OutputEncoding = [System.Text.UTF8Encoding]::new(`$false)
[Console]::InputEncoding = [System.Text.UTF8Encoding]::new(`$false)
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new(`$false)
`$env:PYTHONIOENCODING = 'utf-8'
`$env:PYTHONUTF8 = '1'
& '$python' '$script'
"@

Start-Process `
  -FilePath $powershell `
  -WorkingDirectory $projectRoot `
  -ArgumentList @(
    "-NoExit",
    "-Command",
    $bootstrap
  )

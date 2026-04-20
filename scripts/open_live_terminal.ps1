$ErrorActionPreference = "Stop"
$projectRoot = "C:\3d-recon-pipeline"
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
$script = Join-Path $projectRoot "scripts\watch_live.py"

if (!(Test-Path $python)) {
  throw "找不到 Python：$python"
}

if (!(Test-Path $script)) {
  throw "找不到 watcher：$script"
}

Start-Process `
  -FilePath "C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe" `
  -WorkingDirectory $projectRoot `
  -ArgumentList @(
    "-NoExit",
    "-Command",
    "& '$python' '$script'"
  )

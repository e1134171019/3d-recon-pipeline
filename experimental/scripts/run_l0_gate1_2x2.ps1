param(
    [string]$ImgDir = "C:\3d-recon-pipeline\data\frames_1600",
    [int]$WindowSize = 6,
    [int]$KeepTopK = 1,
    [int]$MinPoints3d = 1,
    [string]$SemanticModelPath = "C:\3d-recon-pipeline\outputs\experiments\l0_semantic_roi\bootstrap36_split_punch_holders_smoke_20260418_005626\weights\best.pt",
    [double]$SemanticConf = 0.25
)

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::UTF8
$OutputEncoding = [Console]::OutputEncoding
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"

$repoRoot = Split-Path -Parent $PSScriptRoot
$pythonExe = Join-Path $repoRoot ".venv\Scripts\python.exe"
$teeScript = Join-Path $repoRoot "scripts\run_and_tee.py"
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$compareRoot = Join-Path $repoRoot "outputs\experiments\l0_selection_compare\l0_2x2_$timestamp"
$logRoot = Join-Path $compareRoot "logs"

New-Item -ItemType Directory -Force -Path $compareRoot | Out-Null
New-Item -ItemType Directory -Force -Path $logRoot | Out-Null

$runs = @(
    @{ Name = "H_D0"; RoiMode = "heuristic"; UseDuplicate = $false },
    @{ Name = "H_D1"; RoiMode = "heuristic"; UseDuplicate = $true  },
    @{ Name = "S_D0"; RoiMode = "semantic";  UseDuplicate = $false },
    @{ Name = "S_D1"; RoiMode = "semantic";  UseDuplicate = $true  }
)

Write-Host "L0 Gate 1 2x2 Compare"
Write-Host "ImgDir            : $ImgDir"
Write-Host "WindowSize        : $WindowSize"
Write-Host "KeepTopK          : $KeepTopK"
Write-Host "MinPoints3d       : $MinPoints3d"
Write-Host "SemanticModelPath : $SemanticModelPath"
Write-Host "SemanticConf      : $SemanticConf"
Write-Host "CompareRoot       : $compareRoot"
Write-Host ""

foreach ($run in $runs) {
    $name = $run.Name
    $runRoot = Join-Path $compareRoot $name
    $logPath = Join-Path $logRoot "$name.log"
    $duplicateFlag = if ($run.UseDuplicate) { "--use-duplicate-penalty" } else { "--no-use-duplicate-penalty" }

    Write-Host "=== Running $name ==="
    $cmd = @(
        $pythonExe,
        "-u",
        "-m",
        "src.run_l0_windowed_selection",
        "--imgdir", $ImgDir,
        "--run-root", $runRoot,
        "--window-size", "$WindowSize",
        "--keep-top-k", "$KeepTopK",
        "--roi-mode", $run.RoiMode,
        "--semantic-model-path", $SemanticModelPath,
        "--semantic-conf", "$SemanticConf",
        $duplicateFlag,
        "--min-points3d", "$MinPoints3d"
    )

    & $pythonExe $teeScript --log $logPath -- $cmd
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
}

Write-Host ""
Write-Host "[OK] L0 Gate 1 2x2 compare finished: $compareRoot"
exit 0

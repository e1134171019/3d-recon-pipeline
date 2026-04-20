param(
    [string]$SubsetDir = "",
    [string]$RunRoot = ""
)

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::UTF8
$OutputEncoding = [Console]::OutputEncoding
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"

$repoRoot = Split-Path -Parent $PSScriptRoot
$pythonExe = Join-Path $repoRoot ".venv\Scripts\python.exe"
$teeScript = Join-Path $repoRoot "scripts\run_and_tee.py"

if (-not (Test-Path -LiteralPath $pythonExe)) {
    throw "找不到 Python: $pythonExe"
}

if ([string]::IsNullOrWhiteSpace($SubsetDir)) {
    $SubsetDir = Join-Path $repoRoot "outputs\experiments\l0_fast_validation\subsets\frames_1600_every6"
}

if (-not (Test-Path -LiteralPath $SubsetDir)) {
    throw "找不到 subset dir: $SubsetDir"
}

if ([string]::IsNullOrWhiteSpace($RunRoot)) {
    $timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $RunRoot = Join-Path $repoRoot "outputs\experiments\l0_fast_validation\gate1_subset_$timestamp"
}

$baselineRoot = Join-Path $RunRoot "baseline_subset"
$baselineWork = Join-Path $baselineRoot "SfM_models\sift"
$baselineLog = Join-Path $baselineRoot "logs\sfm.log"

$a2Root = Join-Path $RunRoot "A2_machine_roi_subset"
$a2Log = Join-Path $a2Root "logs\route.log"

New-Item -ItemType Directory -Force -Path $baselineRoot, $a2Root | Out-Null

Write-Host "Gate 1 subset validation"
Write-Host "Subset : $SubsetDir"
Write-Host "Run    : $RunRoot"
Write-Host ""

$baselineCmd = @(
    $pythonExe,
    "-u",
    "-m",
    "src.sfm_colmap",
    "--imgdir", $SubsetDir,
    "--work", $baselineWork,
    "--max-features", "8192",
    "--seq-overlap", "10",
    "--no-loop-detection",
    "--sift-peak-threshold", "0.006666666666666667",
    "--sift-edge-threshold", "10",
    "--min-points3d", "1"
)

Write-Host "[1/2] baseline subset SfM"
& $pythonExe $teeScript --log $baselineLog -- $baselineCmd
if ($LASTEXITCODE -ne 0) {
    throw "baseline subset SfM 失敗，exit code=$LASTEXITCODE"
}

$a2Cmd = @(
    $pythonExe,
    "-u",
    "-m",
    "src.run_mask_route_opencv",
    "--variant", "A2_machine_roi",
    "--imgdir", $SubsetDir,
    "--run-root", $a2Root,
    "--skip-train",
    "--min-points3d", "1"
)

Write-Host ""
Write-Host "[2/2] A2_machine_roi subset SfM"
& $pythonExe $teeScript --log $a2Log -- $a2Cmd
if ($LASTEXITCODE -ne 0) {
    throw "A2_machine_roi subset 失敗，exit code=$LASTEXITCODE"
}

Write-Host ""
Write-Host "Gate 1 subset validation finished."
Write-Host "Baseline log : $baselineLog"
Write-Host "A2 log       : $a2Log"

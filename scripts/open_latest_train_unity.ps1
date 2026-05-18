param(
    [string]$ContractPath = "C:\3d-recon-pipeline\outputs\reports\agent_train_complete.json",
    [string]$UnityProject = "C:\Users\User\Downloads\phase0\Unity\BendViewer",
    [string]$UnityExe = "C:\Program Files\Unity\Hub\Editor\6000.3.9f1\Editor\Unity.exe",
    [string]$PythonExe = "C:\3d-recon-pipeline\.venv\Scripts\python.exe",
    [string]$AssetBaseName = "",
    [switch]$DryRun,
    [switch]$NoUnityLaunch
)

$ErrorActionPreference = "Stop"

$repoRoot = "C:\3d-recon-pipeline"
$defaultReportContract = Join-Path $repoRoot "outputs\reports\agent_train_complete.json"
$latestEventContract = Join-Path $repoRoot "outputs\agent_events\latest_train_complete.json"
$unityImportScript = Join-Path $repoRoot "scripts\run_unity_batch_import.ps1"

function Resolve-FirstExistingPath {
    param(
        [string[]]$Candidates
    )

    foreach ($candidate in $Candidates) {
        if ([string]::IsNullOrWhiteSpace($candidate)) {
            continue
        }
        if (Test-Path -LiteralPath $candidate) {
            return (Resolve-Path -LiteralPath $candidate).Path
        }
    }

    return $null
}

function Normalize-AssetBaseName {
    param(
        [string]$Name
    )

    $normalized = $Name -replace '\s+', '_'
    $normalized = $normalized -replace '[^A-Za-z0-9_.-]+', '_'
    $normalized = $normalized.Trim('_')
    if ([string]::IsNullOrWhiteSpace($normalized)) {
        return "point_cloud_unity"
    }
    return $normalized
}

$contractCandidates = @()
if (-not [string]::IsNullOrWhiteSpace($ContractPath)) {
    $contractCandidates += $ContractPath
}
if ($ContractPath -eq $defaultReportContract) {
    $contractCandidates += $latestEventContract
}

$resolvedContract = Resolve-FirstExistingPath -Candidates $contractCandidates
if (-not $resolvedContract) {
    throw "找不到 train contract。請確認 $ContractPath 是否存在，或改用最新的 outputs\reports\agent_train_complete.json。"
}

if (-not (Test-Path -LiteralPath $unityImportScript)) {
    throw "找不到 Unity batch import 腳本：$unityImportScript"
}

if (-not (Test-Path -LiteralPath $UnityProject)) {
    throw "找不到 Unity 專案：$UnityProject"
}

$contract = Get-Content -LiteralPath $resolvedContract -Raw | ConvertFrom-Json
$artifacts = $contract.artifacts
$params = $contract.params

if (-not $artifacts.checkpoint) {
    throw "contract 缺少 artifacts.checkpoint：$resolvedContract"
}
if (-not $artifacts.result_dir) {
    throw "contract 缺少 artifacts.result_dir：$resolvedContract"
}

$checkpoint = (Resolve-Path -LiteralPath $artifacts.checkpoint).Path
$resultDir = (Resolve-Path -LiteralPath $artifacts.result_dir).Path
$sceneDir = Join-Path $resultDir "_colmap_scene"

if (-not (Test-Path -LiteralPath $checkpoint)) {
    throw "找不到 checkpoint：$checkpoint"
}
if (-not (Test-Path -LiteralPath $sceneDir)) {
    throw "找不到 COLMAP scene 目錄：$sceneDir"
}

if ([string]::IsNullOrWhiteSpace($AssetBaseName)) {
    $AssetBaseName = Normalize-AssetBaseName -Name ((Split-Path -Leaf $resultDir) + "_point_cloud_unity")
}
else {
    $AssetBaseName = Normalize-AssetBaseName -Name $AssetBaseName
}

$unityExportDir = Join-Path $resultDir "unity_view"
$outputPly = Join-Path $unityExportDir "point_cloud_unity.ply"
$unityLog = Join-Path $unityExportDir ($AssetBaseName + "_unity_import.log")

Write-Host "=== Open Latest Train In Unity ==="
Write-Host "Contract   : $resolvedContract"
Write-Host "Checkpoint  : $checkpoint"
Write-Host "Result Dir  : $resultDir"
Write-Host "Scene Dir   : $sceneDir"
Write-Host "Output PLY  : $outputPly"
Write-Host "Asset Name  : $AssetBaseName"
Write-Host "Unity Proj  : $UnityProject"
Write-Host "Unity Exe   : $UnityExe"
Write-Host "Dry Run     : $DryRun"
Write-Host "Skip GUI    : $NoUnityLaunch"
Write-Host ""

if ($DryRun) {
    Write-Host "[DRYRUN] 會執行："
    Write-Host "  1. python -m src.export_ply_unity --ckpt ... --data-dir ... --out ... --unity"
    Write-Host "  2. scripts/run_unity_batch_import.ps1"
    Write-Host "  3. 啟動 Unity GUI（除非指定 -NoUnityLaunch）"
    exit 0
}

if (-not (Test-Path -LiteralPath $PythonExe)) {
    throw "找不到 Python 執行檔：$PythonExe"
}

Push-Location $repoRoot
try {
    $exportArgs = @(
        "-m", "src.export_ply_unity",
        "--ckpt", $checkpoint,
        "--data-dir", $sceneDir,
        "--out", $outputPly,
        "--unity"
    )

    New-Item -ItemType Directory -Force -Path $unityExportDir | Out-Null

    Write-Host "=== Export PLY for Unity ==="
    & $PythonExe @exportArgs
    if ($LASTEXITCODE -ne 0) {
        throw "export_ply_unity 失敗，exit code = $LASTEXITCODE"
    }

    if (-not (Test-Path -LiteralPath $outputPly)) {
        throw "PLY 匯出後不存在：$outputPly"
    }

    Write-Host ""
    Write-Host "=== Unity Batch Import ==="
    & $unityImportScript -SourcePly $outputPly -UnityProject $UnityProject -LogPath $unityLog -AssetBaseName $AssetBaseName

    if (-not $NoUnityLaunch) {
        if (-not (Test-Path -LiteralPath $UnityExe)) {
            throw "找不到 Unity 執行檔：$UnityExe"
        }

        $unityRunning = Get-Process -Name Unity -ErrorAction SilentlyContinue
        if ($unityRunning) {
            Write-Host "[WARN] 偵測到 Unity 已在執行，略過新視窗啟動。"
        }
        else {
            Write-Host ""
            Write-Host "=== Launch Unity GUI ==="
            Start-Process -FilePath $UnityExe -ArgumentList "-projectPath", $UnityProject
            Write-Host "[OK] Unity Editor 已啟動"
        }
    }
    else {
        Write-Host "[OK] 已完成匯出與 batch import，依照 -NoUnityLaunch 略過 GUI 啟動。"
    }
}
finally {
    Pop-Location
}
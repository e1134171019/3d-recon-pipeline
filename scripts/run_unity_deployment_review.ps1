param(
    [string]$UnityProject = "C:\Users\User\Downloads\phase0\Unity\BendViewer",
    [string]$UnityExe = "C:\Program Files\Unity\Hub\Editor\6000.3.9f1\Editor\Unity.exe",
    [string]$OutputDir = "C:\3d-recon-pipeline\outputs\unity_deployment_review\latest",
    [string]$ScenePath = "Assets/Scenes/FactoryGaussian.unity",
    [string]$SourcePly = "",
    [string]$AssetBaseName = "deployment_review_point_cloud_unity",
    [int]$Width = 1280,
    [int]$Height = 720,
    [int]$TimeoutSec = 600,
    [switch]$FailOnGateFail
)

$ErrorActionPreference = "Stop"

$repoRoot = "C:\3d-recon-pipeline"
$sourceEditorScript = Join-Path $repoRoot "unity_setup\CaptureUnityDeploymentReview.cs"
$unityImportScript = Join-Path $repoRoot "scripts\run_unity_batch_import.ps1"
$targetEditorDir = Join-Path $UnityProject "Assets\Editor"
$targetEditorScript = Join-Path $targetEditorDir "CaptureUnityDeploymentReview.cs"
$importLogPath = Join-Path $OutputDir "unity_import.log"
$logPath = Join-Path $OutputDir "unity_deployment_review.log"
$scorePath = Join-Path $OutputDir "deployment_review_score.json"

if (-not (Test-Path -LiteralPath $UnityExe)) {
    throw "Unity executable not found: $UnityExe"
}
if (-not (Test-Path -LiteralPath $UnityProject)) {
    throw "Unity project not found: $UnityProject"
}
if (-not (Test-Path -LiteralPath $sourceEditorScript)) {
    throw "Review editor script not found: $sourceEditorScript"
}
if (-not [string]::IsNullOrWhiteSpace($SourcePly)) {
    if (-not (Test-Path -LiteralPath $SourcePly)) {
        throw "Source PLY not found: $SourcePly"
    }
    if (-not (Test-Path -LiteralPath $unityImportScript)) {
        throw "Unity batch import script not found: $unityImportScript"
    }
}

$running = Get-Process -Name Unity -ErrorAction SilentlyContinue
if ($running) {
    throw "Unity is already running. Close Unity before batch deployment review."
}

New-Item -ItemType Directory -Force -Path $targetEditorDir | Out-Null
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
Copy-Item -LiteralPath $sourceEditorScript -Destination $targetEditorScript -Force

Write-Host "=== Unity Deployment Review ==="
Write-Host "Unity project : $UnityProject"
Write-Host "Unity exe     : $UnityExe"
Write-Host "Scene         : $ScenePath"
Write-Host "Output dir    : $OutputDir"
Write-Host "Resolution    : ${Width}x${Height}"
Write-Host "Log path      : $logPath"
Write-Host "Source PLY    : $SourcePly"
Write-Host "Asset name    : $AssetBaseName"
Write-Host ""

if (-not [string]::IsNullOrWhiteSpace($SourcePly)) {
    Write-Host "=== Import source PLY before review ==="
    & $unityImportScript -SourcePly $SourcePly -UnityProject $UnityProject -LogPath $importLogPath -AssetBaseName $AssetBaseName
    if ($LASTEXITCODE -ne 0) {
        throw "Unity batch import failed before deployment review: exit code $LASTEXITCODE"
    }
    Write-Host ""
}

$argList = @(
    "-batchmode",
    "-force-d3d12",
    "-projectPath", $UnityProject,
    "-executeMethod", "CaptureUnityDeploymentReview.Run",
    "-reviewOutputDir", $OutputDir,
    "-reviewScene", $ScenePath,
    "-reviewWidth", "$Width",
    "-reviewHeight", "$Height",
    "-quit",
    "-logFile", $logPath
)

$proc = [System.Diagnostics.Process]::Start($UnityExe, [string]::Join(" ", ($argList | ForEach-Object {
    if ($_ -match '\s') { '"' + ($_ -replace '"', '\"') + '"' } else { $_ }
})))
if (-not $proc) {
    throw "Failed to start Unity process"
}

if (-not $proc.WaitForExit($TimeoutSec * 1000)) {
    try {
        $proc.Kill($true)
    }
    catch {
        $proc.Kill()
    }
    throw "Unity deployment review timed out after $TimeoutSec seconds"
}

$exitCode = $proc.ExitCode
Write-Host "Unity exit code: $exitCode"

if (Test-Path -LiteralPath $logPath) {
    Write-Host ""
    Write-Host "=== Unity review log tail ==="
    Get-Content -LiteralPath $logPath -Tail 100
}
else {
    Write-Host "[WARN] Unity log not found: $logPath"
}

if ($exitCode -ne 0) {
    throw "Unity deployment review failed with exit code $exitCode"
}

if (-not (Test-Path -LiteralPath $scorePath)) {
    throw "Deployment review score was not created: $scorePath"
}

$score = Get-Content -LiteralPath $scorePath -Raw | ConvertFrom-Json

Write-Host ""
Write-Host "=== Deployment review score ==="
Write-Host "Score path            : $scorePath"
Write-Host "Import success        : $($score.import_success)"
Write-Host "Views evaluated       : $($score.metrics.views_evaluated)"
Write-Host "Valid views           : $($score.metrics.valid_views)"
Write-Host "White haze mean       : $($score.metrics.white_haze_ratio_mean)"
Write-Host "Bright clip mean      : $($score.metrics.bright_clip_ratio_mean)"
Write-Host "Dark void mean        : $($score.metrics.dark_void_ratio_mean)"
Write-Host "Edge sharpness mean   : $($score.metrics.edge_sharpness_mean)"
Write-Host "Deployment pass       : $($score.deployment_review_pass)"
Write-Host "Failure tags          : $($score.failure_tags -join ', ')"

if ($FailOnGateFail -and -not [bool]$score.deployment_review_pass) {
    throw "Deployment review gate failed"
}

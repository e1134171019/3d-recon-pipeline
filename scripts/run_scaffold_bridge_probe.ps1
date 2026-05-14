param(
    [Parameter(Mandatory = $true)][string]$ModelPath,
    [string]$SandboxRoot = "C:\3d-recon-pipeline\experimental\scaffold_gs_probe",
    [string]$UnityProject = "",
    [int]$Iteration = -1,
    [ValidateSet("test", "train")]
    [string]$CameraSet = "test",
    [int]$CameraIndex = 0,
    [int]$MaxSplats = 300000,
    [double]$MinOpacity = 0.0,
    [string]$AssetBaseName = "",
    [string]$Gpu = "0",
    [switch]$NoUnityCoordinates
)

$ErrorActionPreference = "Stop"

$repoRoot = "C:\3d-recon-pipeline"
$venvPython = Join-Path $SandboxRoot ".venv_scaffold\Scripts\python.exe"
$exportScript = Join-Path $SandboxRoot "export_scaffold_gs_unity_ply.py"
$schemaPath = Join-Path $SandboxRoot "bridge_score_schema.json"
$unityImportScript = Join-Path $repoRoot "scripts\run_unity_batch_import.ps1"
$modelPathResolved = (Resolve-Path -LiteralPath $ModelPath).Path

if (-not (Test-Path -LiteralPath $venvPython)) {
    throw "Sandbox venv not found: $venvPython"
}
if (-not (Test-Path -LiteralPath $exportScript)) {
    throw "Export script not found: $exportScript"
}
if (-not (Test-Path -LiteralPath $schemaPath)) {
    throw "Bridge score schema not found: $schemaPath"
}
if (-not (Test-Path -LiteralPath $unityImportScript)) {
    throw "Unity batch import script not found: $unityImportScript"
}

$unityExportsDir = Join-Path $modelPathResolved "unity_exports"
New-Item -ItemType Directory -Force -Path $unityExportsDir | Out-Null

if ([string]::IsNullOrWhiteSpace($AssetBaseName)) {
    $leaf = Split-Path -Leaf $modelPathResolved
    $leafShort = if ($leaf.Length -gt 24) { $leaf.Substring(0, 24) } else { $leaf }
    $AssetBaseName = "sgs_${leafShort}_v${CameraIndex}_${([int]($MaxSplats / 1000))}k"
    $AssetBaseName = $AssetBaseName.Replace("-", "_")
}

$outputPly = Join-Path $unityExportsDir ($AssetBaseName + ".ply")
$exportReport = Join-Path $unityExportsDir ($AssetBaseName + "_export_report.json")
$bridgeScore = Join-Path $unityExportsDir ($AssetBaseName + "_bridge_score.json")
$unityLog = Join-Path $unityExportsDir ($AssetBaseName + "_unity_import.log")
$unityAssetPath = ""

Write-Host "=== Scaffold Bridge Probe ==="
Write-Host "Model Path    : $modelPathResolved"
Write-Host "Camera        : $CameraSet / $CameraIndex"
Write-Host "Output PLY    : $outputPly"
Write-Host "Export Report : $exportReport"
Write-Host "Bridge Score  : $bridgeScore"
if (-not [string]::IsNullOrWhiteSpace($UnityProject)) {
    Write-Host "Unity Project : $UnityProject"
}
Write-Host ""

$exportArgs = @(
    $exportScript,
    "-m", $modelPathResolved,
    "--iteration", $Iteration,
    "--camera-set", $CameraSet,
    "--camera-index", $CameraIndex,
    "--max-splats", $MaxSplats,
    "--min-opacity", $MinOpacity,
    "--report-output", $exportReport,
    "--output", $outputPly,
    "--gpu", $Gpu
)
if ($NoUnityCoordinates) {
    $exportArgs += "--no-unity-coordinates"
}

$exportOk = $false
$exportError = ""
try {
    & $venvPython @exportArgs
    if ($LASTEXITCODE -eq 0) {
        $exportOk = $true
    }
    else {
        $exportError = "Export script exited with code $LASTEXITCODE"
    }
}
catch {
    $exportError = $_.Exception.Message
}

$template = Get-Content -LiteralPath $schemaPath -Raw | ConvertFrom-Json
$template.source_run = $modelPathResolved
$template.iteration = $Iteration
$template.export.camera_set = $CameraSet
$template.export.camera_index = $CameraIndex
$template.export.min_opacity = $MinOpacity
$template.export.max_splats = $MaxSplats
$template.export.output_ply = $outputPly
$template.export.report_json = $exportReport
$template.export.export_ok = $false
$template.export.export_error = $exportError

if ($exportOk -and (Test-Path -LiteralPath $exportReport)) {
    $exportData = Get-Content -LiteralPath $exportReport -Raw | ConvertFrom-Json
    $template.export.candidate_splats = $exportData.candidate_splats
    $template.export.exported_splats = $exportData.exported_splats
    $template.export.output_size_mb = $exportData.output_size_mb
    $template.export.export_ok = [bool](Test-Path -LiteralPath $outputPly)
}

$template.gate.export_gate_pass = [bool]$template.export.export_ok

$importOk = $false
$successMarkerFound = $false
$importError = ""

if ($template.gate.export_gate_pass -and -not [string]::IsNullOrWhiteSpace($UnityProject)) {
    try {
        & $unityImportScript -SourcePly $outputPly -UnityProject $UnityProject -LogPath $unityLog -AssetBaseName $AssetBaseName
        $unityAssetPath = Join-Path $UnityProject "Assets\GaussianAssets\$AssetBaseName.asset"
        if (Test-Path -LiteralPath $unityLog) {
            $successMarkerFound = [bool](Select-String -LiteralPath $unityLog -SimpleMatch "[GaussianSplatBatchImport] 匯入完成：" -Quiet)
        }
        $importOk = $successMarkerFound -or (Test-Path -LiteralPath $unityAssetPath)
    }
    catch {
        $importError = $_.Exception.Message
        $unityAssetPath = Join-Path $UnityProject "Assets\GaussianAssets\$AssetBaseName.asset"
    }
}

$template.unity_import.unity_project = $UnityProject
$template.unity_import.asset_base_name = $AssetBaseName
$template.unity_import.log_path = $unityLog
$template.unity_import.asset_path = $unityAssetPath
$template.unity_import.import_ok = $importOk
$template.unity_import.success_marker_found = $successMarkerFound
$template.unity_import.import_error = $importError
$template.gate.import_gate_pass = $importOk

if ($template.gate.export_gate_pass -and $template.gate.import_gate_pass) {
    $template.gate.reason = "Awaiting deployment-side visual review"
}
elseif (-not $template.gate.export_gate_pass) {
    $template.gate.reason = if ($exportError) { "Export failed: $exportError" } else { "Export failed" }
}
else {
    $template.gate.reason = if ($importError) { "Unity import failed: $importError" } else { "Unity import not run or not confirmed" }
}

$template | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $bridgeScore -Encoding utf8

Write-Host ""
Write-Host "=== Bridge Probe Result ==="
Write-Host "Export Gate Pass : $($template.gate.export_gate_pass)"
Write-Host "Import Gate Pass : $($template.gate.import_gate_pass)"
Write-Host "Bridge Pass      : $($template.gate.bridge_pass)"
Write-Host "Reason           : $($template.gate.reason)"
Write-Host "Score JSON       : $bridgeScore"

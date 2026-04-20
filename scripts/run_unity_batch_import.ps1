param(
    [Parameter(Mandatory = $true)][string]$SourcePly,
    [Parameter(Mandatory = $true)][string]$UnityProject,
    [Parameter(Mandatory = $true)][string]$LogPath,
    [string]$AssetBaseName = "point_cloud_unity"
)

$ErrorActionPreference = "Stop"

$UnityExe = "C:\Program Files\Unity\Hub\Editor\6000.3.9f1\Editor\Unity.exe"
$DestDir = Join-Path $UnityProject "Assets\GaussianSplats"
$DestPly = Join-Path $DestDir ($AssetBaseName + ".ply")
$AssetDir = Join-Path $UnityProject "Assets\GaussianAssets"
$SuccessMarker = "[GaussianSplatBatchImport] 匯入完成：Assets/GaussianAssets/$AssetBaseName.asset"

New-Item -ItemType Directory -Force -Path $DestDir | Out-Null
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $LogPath) | Out-Null

Write-Host "=== Unity Batch Import ==="
Write-Host "Source PLY : $SourcePly"
Write-Host "Unity Proj : $UnityProject"
Write-Host "Log Path   : $LogPath"
Write-Host "Asset Name : $AssetBaseName"
Write-Host "Graphics   : D3D12 (graphics enabled)"
Write-Host ""

Copy-Item -LiteralPath $SourcePly -Destination $DestPly -Force
Write-Host "[OK] Copied to $DestPly"

$argList = @(
    "-batchmode"
    "-force-d3d12"
    "-projectPath", $UnityProject
    "-executeMethod", "GaussianSplatBatchImport.ImportPointCloudUnityByName"
    "-assetBaseName", $AssetBaseName
    "-quit"
    "-logFile", $LogPath
)

$proc = Start-Process -FilePath $UnityExe -ArgumentList $argList -Wait -PassThru -NoNewWindow
$ExitCode = $proc.ExitCode

Write-Host ""
Write-Host "=== Unity import log ==="
if (Test-Path -LiteralPath $LogPath) {
    Get-Content -LiteralPath $LogPath -Tail 120
} else {
    Write-Host "[WARN] Log file not found."
}

Write-Host ""
Write-Host "=== Imported asset check ==="
if (Test-Path -LiteralPath $AssetDir) {
    Get-ChildItem -LiteralPath $AssetDir | Select-Object Name, Length, LastWriteTime
} else {
    Write-Host "[WARN] Asset dir not found: $AssetDir"
}

$ImportSucceeded = $false
if (Test-Path -LiteralPath $LogPath) {
    $ImportSucceeded = [bool](Select-String -LiteralPath $LogPath -SimpleMatch $SuccessMarker -Quiet)
}

if ($ExitCode -ne 0 -and -not $ImportSucceeded) {
    throw "Unity batch import failed with exit code $ExitCode"
}

Write-Host ""
if ($ImportSucceeded) {
    Write-Host "[OK] Unity batch import finished."
} else {
    Write-Host "[WARN] Unity exited without success marker. Please inspect the log."
}

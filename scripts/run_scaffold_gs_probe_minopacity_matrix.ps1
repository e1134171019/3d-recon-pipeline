param(
    [string]$SandboxRoot = "C:\3d-recon-pipeline\experimental\scaffold_gs_probe",
    [string]$DatasetName = "factorygaussian",
    [string]$SceneName = "u_base_750k_aa",
    [string]$Gpu = "0",
    [int]$Iterations = 7000,
    [double]$VoxelSize = 0.01,
    [int]$Ratio = 5,
    [int]$Resolution = 8,
    [int]$AppearanceDim = 0,
    [double]$LambdaDssim = 0.3,
    [int[]]$OnlyIndex,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

$entries = @(
    @{ Name = "scaffold_minop003_r5_res8_v0010_app0_dssim03_7k"; MinOpacity = 0.003 },
    @{ Name = "scaffold_minop005_r5_res8_v0010_app0_dssim03_7k"; MinOpacity = 0.005 },
    @{ Name = "scaffold_minop010_r5_res8_v0010_app0_dssim03_7k"; MinOpacity = 0.01 }
)

if ($OnlyIndex -and $OnlyIndex.Count -gt 0) {
    $selected = @()
    foreach ($idx in $OnlyIndex) {
        if ($idx -lt 1 -or $idx -gt $entries.Count) {
            throw "OnlyIndex out of range: $idx"
        }
        $selected += $entries[$idx - 1]
    }
    $entries = $selected
}

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$runner = "C:\3d-recon-pipeline\scripts\run_scaffold_gs_probe_windows.ps1"

Write-Host "[MATRIX] Scaffold-GS min_opacity sweep"
Write-Host "Fixed: ratio=$Ratio resolution=$Resolution voxel_size=$VoxelSize appearance_dim=$AppearanceDim lambda_dssim=$LambdaDssim iterations=$Iterations"

$i = 1
foreach ($entry in $entries) {
    $expName = "{0}_{1}" -f $entry.Name, $timestamp
    Write-Host ("[{0}] {1} min_opacity={2}" -f $i, $expName, $entry.MinOpacity)
    if (-not $DryRun) {
        & $runner `
            -SandboxRoot $SandboxRoot `
            -DatasetName $DatasetName `
            -SceneName $SceneName `
            -ExperimentName $expName `
            -Gpu $Gpu `
            -Iterations $Iterations `
            -VoxelSize $VoxelSize `
            -Ratio $Ratio `
            -Resolution $Resolution `
            -AppearanceDim $AppearanceDim `
            -LambdaDssim $LambdaDssim `
            -MinOpacity $entry.MinOpacity `
            -DataDevice cpu
    }
    $i += 1
}

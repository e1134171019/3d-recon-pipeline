param(
    [string]$SandboxRoot = "C:\3d-recon-pipeline\experimental\scaffold_gs_probe",
    [string]$DatasetName = "factorygaussian",
    [string]$SceneName = "u_base_750k_aa",
    [ValidateSet("density_only", "appearance_only")]
    [string]$Mode,
    [string]$Gpu = "0",
    [int]$Iterations = 7000
)

$ErrorActionPreference = "Stop"

$runner = Join-Path $PSScriptRoot "run_scaffold_gs_probe_windows.ps1"
if (-not (Test-Path -LiteralPath $runner)) {
    throw "Runner missing: $runner"
}

# Baseline is the current best-known Scaffold sandbox branch:
# ratio=5, resolution=8, voxel_size=0.01, appearance_dim=0,
# lambda_dssim=0.3, min_opacity=0.003.
$commonArgs = @{
    SandboxRoot   = $SandboxRoot
    DatasetName   = $DatasetName
    SceneName     = $SceneName
    Gpu           = $Gpu
    Iterations    = $Iterations
    LambdaDssim   = 0.3
    MinOpacity    = 0.003
    DataDevice    = "cuda"
}

switch ($Mode) {
    "density_only" {
        $exp = "split_density_r1_resm1_v0001_app0_d03_m003_${Iterations}i"
        $args = @{
            ExperimentName = $exp
            Ratio          = 1
            Resolution     = -1
            VoxelSize      = 0.001
            AppearanceDim  = 0
        }
    }
    "appearance_only" {
        $exp = "split_app_r5_res8_v0010_app32_d03_m003_${Iterations}i"
        $args = @{
            ExperimentName = $exp
            Ratio          = 5
            Resolution     = 8
            VoxelSize      = 0.01
            AppearanceDim  = 32
        }
    }
}

Write-Host "[SPLIT] Scaffold-GS probe"
Write-Host "Mode: $Mode"
Write-Host "Experiment: $exp"

& $runner @commonArgs @args

param(
    [string]$SandboxRoot = "C:\3d-recon-pipeline\experimental\scaffold_gs_probe",
    [string]$DatasetName = "factorygaussian",
    [string]$SceneName = "u_base_750k_aa",
    [ValidateSet("step1_ratio1_res8_v0010", "step2_ratio5_resm1_v0010", "step3_ratio1_resm1_v0020")]
    [string]$Step = "step1_ratio1_res8_v0010",
    [string]$Suffix = "",
    [string]$Gpu = "0",
    [int]$Port = 6011,
    [int]$Iterations = 7000
)

$ErrorActionPreference = "Stop"

$runner = Join-Path $PSScriptRoot "run_scaffold_gs_probe_windows.ps1"
if (-not (Test-Path -LiteralPath $runner)) {
    throw "Runner missing: $runner"
}

# Fixed Scaffold sandbox baseline:
# ratio=5, resolution=8, voxel_size=0.01, appearance_dim=0,
# lambda_dssim=0.3, min_opacity=0.003, data_device=cuda.
$commonArgs = @{
    SandboxRoot  = $SandboxRoot
    DatasetName  = $DatasetName
    SceneName    = $SceneName
    Gpu          = $Gpu
    Port         = $Port
    Iterations   = $Iterations
    AppearanceDim = 0
    LambdaDssim  = 0.3
    MinOpacity   = 0.003
    DataDevice   = "cuda"
}

switch ($Step) {
    "step1_ratio1_res8_v0010" {
        $experimentName = "ladder_step1_r1_res8_v0010_app0_d03_m003_${Iterations}i"
        $stepArgs = @{
            ExperimentName = $experimentName
            Ratio          = 1
            Resolution     = 8
            VoxelSize      = 0.01
        }
        $summary = "Only lift ratio to 1; keep sandbox resolution and voxel."
    }
    "step2_ratio5_resm1_v0010" {
        $experimentName = "ladder_step2_r5_resm1_v0010_app0_d03_m003_${Iterations}i"
        $stepArgs = @{
            ExperimentName = $experimentName
            Ratio          = 5
            Resolution     = -1
            VoxelSize      = 0.01
        }
        $summary = "Only lift resolution to full; keep sandbox ratio and voxel."
    }
    "step3_ratio1_resm1_v0020" {
        $experimentName = "ladder_step3_r1_resm1_v0020_app0_d03_m003_${Iterations}i"
        $stepArgs = @{
            ExperimentName = $experimentName
            Ratio          = 1
            Resolution     = -1
            VoxelSize      = 0.02
        }
        $summary = "Full-scene density with relaxed voxel to bound anchor growth."
    }
}

if ($Suffix) {
    $experimentName = "${experimentName}_$Suffix"
}
$stepArgs["ExperimentName"] = $experimentName

Write-Host "[LADDER] Scaffold density diagnostic"
Write-Host "Step: $Step"
Write-Host "Experiment: $experimentName"
Write-Host "Summary: $summary"

& $runner @commonArgs @stepArgs

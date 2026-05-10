## Scaffold-GS Probe Sandbox

This sandbox is isolated from the formal `gsplat` mainline.

### Purpose

- Evaluate whether `Scaffold-GS` can reuse existing `_colmap_scene/images + sparse/0`
- Keep all probe artifacts outside the formal training pipeline
- Preserve a clean rollback path if dependency or format issues block execution

### Expected layout

```text
experimental/scaffold_gs_probe/
├── repo/                     # cloned Scaffold-GS repo
├── data/
│   └── factorygaussian/
│       └── u_base_750k_aa/
│           ├── images/
│           └── sparse/0/
└── outputs/
```

### Current source scene

Default source scene:

```text
C:\3d-recon-pipeline\outputs\experiments\train_probes\u_base_mcmc_capmax_aa_fulltrain_20260420_032355\mcmc_capmax_750k_aa\_colmap_scene
```

### Guardrails

- Do not modify formal `src/train_3dgs.py`
- Do not modify formal `src/export_ply.py`
- Do not modify formal `src/export_ply_unity.py`
- Do not write Scaffold-GS artifacts into formal `outputs/experiments`

### Offline Teacher Hook

- `C:\3d-recon-pipeline\scripts\run_scaffold_teacher_loop.ps1`
- Builds `Scaffold-GS` sandbox seed records for `D:\agent_test`
- Sends them through local `Ollama/Qwen`
- Optionally merges them with the existing historical teacher dataset and retrains the offline PyTorch baseline

### Known-good sanity probe

The first successful Windows sanity probe used:

```powershell
pwsh -File C:\3d-recon-pipeline\scripts\run_scaffold_gs_probe_windows.ps1 `
  -ExperimentName probe_fast_cpuimg_r8_ratio10_300 `
  -Iterations 300 `
  -VoxelSize 0.01 `
  -Ratio 10 `
  -Resolution 8 `
  -DataDevice cpu `
  -Gpu 0
```

Why these settings matter:

- `ratio=10` reduces initial anchors.
- `resolution=8` reduces image tensor size.
- `data_device=cpu` prevents all camera images from being preloaded into GPU memory.

Do not use full-scene defaults (`data_device=cuda`, `resolution=-1`, `ratio=1`) for quick probes; on the current 853-image scene they can fill VRAM before providing useful signal.

### Known-good 7000-iteration probe

After the sanity probe, the first larger sandbox run used:

```powershell
pwsh -File C:\3d-recon-pipeline\scripts\run_scaffold_gs_probe_windows.ps1 `
  -ExperimentName probe_mid_cpuimg_r4_ratio5_7000 `
  -Iterations 7000 `
  -VoxelSize 0.01 `
  -Ratio 5 `
  -Resolution 4 `
  -DataDevice cpu `
  -Gpu 0
```

Observed result on 2026-05-08:

- `anchors = 110,942`
- `Test FPS = 200.77679`
- `SSIM = 0.8642440`
- `PSNR = 23.9623814`
- `LPIPS = 0.1389390`

This is still a sandbox probe. Do not promote it over the formal `U_base + MCMCStrategy` line until Unity import/render validation is complete.

### Unity import bridge

The raw Scaffold-GS `point_cloud.ply` is not a UnityGaussianSplatting asset
source. It stores anchors, offsets, anchor features, and MLP checkpoints. The
Unity plugin expects explicit INRIA-style Gaussian splats.

Use the sandbox converter for a first importable preview:

```powershell
& C:\3d-recon-pipeline\experimental\scaffold_gs_probe\.venv_scaffold\Scripts\python.exe `
  C:\3d-recon-pipeline\experimental\scaffold_gs_probe\export_scaffold_gs_unity_ply.py `
  -m C:\3d-recon-pipeline\experimental\scaffold_gs_probe\repo\Scaffold-GS\outputs\factorygaussian\u_base_750k_aa\probe_mid_cpuimg_r4_ratio5_7000_20260508_072420 `
  --iteration 7000 `
  --camera-set test `
  --camera-index 0 `
  --max-splats 300000 `
  --output C:\3d-recon-pipeline\experimental\scaffold_gs_probe\repo\Scaffold-GS\outputs\factorygaussian\u_base_750k_aa\probe_mid_cpuimg_r4_ratio5_7000_20260508_072420\unity_exports\scaffold_gs_r4_ratio5_7000_view0_300k.ply
```

The first converted preview exported `256,202` splats and imported into
Unity as `scaffold_gs_r4_ratio5_7000_view0_300k.asset`. Treat this as an
import-compatibility bridge, not a final quality export.

### Recommended 7000-iteration matrix

Use the matrix launcher to accumulate comparable `Scaffold-GS` probe records
without changing the formal mainline:

```powershell
pwsh -File C:\3d-recon-pipeline\scripts\run_scaffold_gs_probe_matrix.ps1
```

Current first-round matrix:

1. `ratio=5`, `resolution=4`, `voxel_size=0.01`
2. `ratio=5`, `resolution=8`, `voxel_size=0.01`
3. `ratio=10`, `resolution=4`, `voxel_size=0.01`
4. `ratio=10`, `resolution=8`, `voxel_size=0.01`
5. `ratio=5`, `resolution=4`, `voxel_size=0.02`
6. `ratio=10`, `resolution=4`, `voxel_size=0.02`

All entries use:

- `iterations=7000`
- `data_device=cpu`
- `gpu=0`

To run only selected entries:

```powershell
pwsh -File C:\3d-recon-pipeline\scripts\run_scaffold_gs_probe_matrix.ps1 `
  -OnlyIndex 1,3,6
```

To verify the matrix naming and parameters without launching training:

```powershell
pwsh -File C:\3d-recon-pipeline\scripts\run_scaffold_gs_probe_matrix.ps1 `
  -DryRun
```

### Current provisional best from round 1

Among the first completed `7000`-iteration matrix entries on `2026-05-09`,
the best current result is:

- `ratio=5`
- `resolution=8`
- `voxel_size=0.01`
- `LPIPS=0.0728270`
- `PSNR=25.3738117`
- `SSIM=0.9132568`

Treat this as a provisional winner until the full round-1 matrix finishes.

### Round 2: appearance_dim sweep

After round 1 completes, the next focused matrix should keep the provisional
best geometry/resource settings fixed and only sweep `appearance_dim`:

```powershell
pwsh -File C:\3d-recon-pipeline\scripts\run_scaffold_gs_probe_appearance_matrix.ps1 `
  -Ratio 5 `
  -Resolution 8 `
  -VoxelSize 0.01
```

Default appearance sweep:

1. `appearance_dim=0`
2. `appearance_dim=16`
3. `appearance_dim=32`

To inspect names and parameters first:

```powershell
pwsh -File C:\3d-recon-pipeline\scripts\run_scaffold_gs_probe_appearance_matrix.ps1 `
  -DryRun
```

Observed result on `2026-05-09`:

1. `appearance_dim=0`
   - `SSIM=0.9131400`
   - `PSNR=25.3483982`
   - `LPIPS=0.0735160`
2. `appearance_dim=16`
   - `SSIM=0.8837507`
   - `PSNR=23.1264973`
   - `LPIPS=0.0888688`
3. `appearance_dim=32`
   - `SSIM=0.8671480`
   - `PSNR=22.4782391`
   - `LPIPS=0.0940574`

Conclusion:

- `appearance_dim=0` is effectively tied with the best round-1 run and remains the current best.
- Increasing appearance capacity to `16/32` hurt both geometry-consistent quality and LPIPS on this scene.

### Round 3: use_feat_bank sweep

Keep the current best settings fixed and only test the binary feature-bank
switch:

```powershell
pwsh -File C:\3d-recon-pipeline\scripts\run_scaffold_gs_probe_featbank_matrix.ps1
```

Fixed baseline:

- `ratio=5`
- `resolution=8`
- `voxel_size=0.01`
- `appearance_dim=0`
- `iterations=7000`
- `data_device=cpu`

Entries:

1. `use_feat_bank=False`
2. `use_feat_bank=True`

To inspect names and parameters first:

```powershell
pwsh -File C:\3d-recon-pipeline\scripts\run_scaffold_gs_probe_featbank_matrix.ps1 `
  -DryRun
```

Observed result on `2026-05-09`:

1. `use_feat_bank=False`
   - `SSIM=0.9132942`
   - `PSNR=25.3544559`
   - `LPIPS=0.0731829`
2. `use_feat_bank=True`
   - invalid under `appearance_dim=0`
   - training failed in `scene/gaussian_model.py` because `embedding_appearance` was `None`

Conclusion:

- `use_feat_bank=False` remains the current best valid setting.
- `use_feat_bank=True` is not a valid next-step sweep under `appearance_dim=0`; revisit only if a later branch proves `appearance_dim>0` is worth paying for.

### Round 4: lambda_dssim sweep

Keep the current best settings fixed and sweep only the DSSIM loss weight:

```powershell
pwsh -File C:\3d-recon-pipeline\scripts\run_scaffold_gs_probe_dssim_matrix.ps1
```

Fixed baseline:

- `ratio=5`
- `resolution=8`
- `voxel_size=0.01`
- `appearance_dim=0`
- `use_feat_bank=False`
- `iterations=7000`
- `data_device=cpu`

Entries:

1. `lambda_dssim=0.1`
2. `lambda_dssim=0.2`
3. `lambda_dssim=0.3`

Observed result on `2026-05-09`:

1. `lambda_dssim=0.1`
   - `SSIM=0.9063165`
   - `PSNR=25.1922588`
   - `LPIPS=0.0801650`
2. `lambda_dssim=0.2`
   - `SSIM=0.9128506`
   - `PSNR=25.2884388`
   - `LPIPS=0.0732566`
3. `lambda_dssim=0.3`
   - `SSIM=0.9169572`
   - `PSNR=25.4211292`
   - `LPIPS=0.0682545`

Conclusion:

- `lambda_dssim=0.3` is the current best overall Scaffold-GS setting.
- This is the first Scaffold-GS probe to clearly beat the earlier `LPIPS ~0.073` plateau.

### Round 5: min_opacity sweep

After locking the current best geometry and loss settings, the next focused
matrix should test densification/prune behavior through `min_opacity`:

```powershell
pwsh -File C:\3d-recon-pipeline\scripts\run_scaffold_gs_probe_minopacity_matrix.ps1
```

Fixed baseline:

- `ratio=5`
- `resolution=8`
- `voxel_size=0.01`
- `appearance_dim=0`
- `use_feat_bank=False`
- `lambda_dssim=0.3`
- `iterations=7000`
- `data_device=cpu`

Entries:

1. `min_opacity=0.003`
2. `min_opacity=0.005`
3. `min_opacity=0.01`

To inspect names and parameters first:

```powershell
pwsh -File C:\3d-recon-pipeline\scripts\run_scaffold_gs_probe_minopacity_matrix.ps1 `
  -DryRun
```

Keep the current best valid settings fixed and only test the image-loss
mixture weight:

```powershell
pwsh -File C:\3d-recon-pipeline\scripts\run_scaffold_gs_probe_dssim_matrix.ps1
```

Fixed baseline:

- `ratio=5`
- `resolution=8`
- `voxel_size=0.01`
- `appearance_dim=0`
- `use_feat_bank=False`
- `iterations=7000`
- `data_device=cpu`

Entries:

1. `lambda_dssim=0.1`
2. `lambda_dssim=0.2`
3. `lambda_dssim=0.3`

To inspect names and parameters first:

```powershell
pwsh -File C:\3d-recon-pipeline\scripts\run_scaffold_gs_probe_dssim_matrix.ps1 `
  -DryRun
```

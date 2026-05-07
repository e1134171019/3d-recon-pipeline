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

# Scaffold-GS Bridge Checklist

This checklist defines the bridge-first milestone for the Scaffold-GS sandbox.
Do not reopen large trainer-level sweeps until this checklist yields a
trustworthy deployment-side signal.

## Scope

- Sandbox only: `experimental/scaffold_gs_probe`
- Export entrypoint: `experimental/scaffold_gs_probe/export_scaffold_gs_unity_ply.py`
- Unity import entrypoint: `scripts/run_unity_batch_import.ps1`
- Output record: `bridge_score_schema.json`

## Fixed baseline before bridge reopens trainer knobs

Keep the current best sandbox branch fixed unless the bridge itself requires a
change:

- `voxel_size = 0.01`
- `appearance_dim = 0`
- `use_feat_bank = False`
- `lambda_dssim = 0.3`
- `min_opacity = 0.003`
- `data_device = cpu`

Do not reopen `ratio` or `resolution` before the bridge score is trustworthy.

## Stage 1: Export gate

Pass conditions:

- Export script finishes without exception
- Export report JSON exists
- `candidate_splats > 0`
- `exported_splats > 0`
- Output PLY exists

Failure examples:

- Missing Scaffold checkpoint path
- Export script throws
- PLY file missing
- Zero exported splats

## Stage 2: Unity import gate

Pass conditions:

- Unity batch import returns without fatal error
- Unity log exists
- Success marker found in Unity log, or expected `.asset` exists
- Imported asset path is recorded in the bridge score JSON

Failure examples:

- Unity batch import exit failure
- No success marker and no asset generated
- Asset count is zero / missing

## Stage 3: Deployment-side visual review

Required manual review cameras:

- Near view
- Mid view
- Far view

Required review fields in the bridge score JSON:

- `near_view_result`
- `mid_view_result`
- `far_view_result`
- `fogging`
- `fragmentation`
- `structure_stability`
- `visual_result`

Pass conditions:

- No catastrophic fogging in near view
- No severe fragmentation / exploded structure
- Major scene structure remains recognizable
- Visual result is at least acceptable for side-by-side comparison

## Stage 4: Bridge-first milestone

`bridge_pass = true` only when all are true:

- Export gate passed
- Unity import gate passed
- Deployment-side visual review passed

If only export/import succeed, record the run as:

- `export_gate_pass = true`
- `import_gate_pass = true`
- `deployment_review_pass = false`
- `bridge_pass = false`
- `reason = "Awaiting deployment-side visual review"`

## What bridge pass unlocks

Only after `bridge_pass = true` may the sandbox reopen:

- `n_offsets`
- `add_color_dist`
- `add_opacity_dist`
- `add_cov_dist`
- Fair `ratio = 1`
- Fair `resolution = -1`

Do not launch the large PB / SHA screen before this milestone.

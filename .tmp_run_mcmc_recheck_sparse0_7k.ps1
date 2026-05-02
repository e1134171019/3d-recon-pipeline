[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()
$OutputEncoding = [Console]::OutputEncoding
$env:PYTHONIOENCODING = 'utf-8'

& 'C:\3d-recon-pipeline\.venv\Scripts\python.exe' -u -m src.train_3dgs `
  --train-mode mcmc `
  --imgdir data/frames_1600 `
  --colmap outputs/SfM_models/sift/sparse/0 `
  --validation-report outputs/reports/pointcloud_validation_report.json `
  --outdir outputs/experiments/mcmc_recheck_legacy_sparse0_7k_retry_20260429 `
  --iterations 7000 `
  --eval-steps 1000 `
  --cap-max 1000000 `
  --disable-video

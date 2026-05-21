# Observer UI

Vue-based read-only dashboard for the 3D recon pipeline.

This layer observes formal artifacts only. It does not control training, Unity, Ollama, PyTorch, or formal decisions.

## Scope

- `slot`: `decision / Unity runtime observability`
- `scope`: `formal-adjacent observer`
- `bottleneck`: terminal output and scattered JSON reports are hard to monitor during long runs.
- `baseline`: manual inspection of logs, `latest_*` JSON files, teacher reports, and learner reports.
- `hypothesis`: a read-only dashboard reduces missed stale artifacts without changing formal runtime behavior.
- `gate`: UI may read formal artifacts; UI may not write `latest_*_decision.json`.
- `success`: `/api/health` and `/api/snapshot` work, and Vue displays current formal runtime, teacher, learner, MCMC, and deployment state.
- `kill`: stop if the UI needs direct decision writes or tries to let Ollama/Qwen control runtime.
- `artifact`: `observer_ui/server.py`, `observer_ui/static/*`, and this README.

## Run

```powershell
cd C:\3d-recon-pipeline
.\.venv\Scripts\python.exe observer_ui\server.py --host 127.0.0.1 --port 8765
```

Then open:

```text
http://127.0.0.1:8765
```

For a non-interactive smoke check:

```powershell
.\.venv\Scripts\python.exe observer_ui\server.py --check
```

## Data Sources

- `outputs/agent_events/latest_*_complete.json`
- `outputs/agent_decisions/latest_*_decision.json`
- `outputs/reports/mcmc_run_inventory.summary.json`
- latest `outputs/experiments/**/deployment_review.json`
- `experimental/scaffold_gs_probe/latest_teacher_loop_status.json`
- `D:\agent_test\outputs\offline_learning\historical_plus_scaffold_report.json`
- `D:\agent_test\outputs\offline_learning\augmented_pytorch_baseline_report.json`

## Guardrail

V0 intentionally has no pass/fail buttons. Future UI actions must call a formal CLI and produce `outcome_feedback` or `deployment_review` artifacts before teacher / learner can absorb the result.

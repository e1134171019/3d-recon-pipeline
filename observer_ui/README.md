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
- `artifact`: `observer_ui/server.py`, `observer_ui/record_activity.py`, `observer_ui/static/*`, `outputs/observer_events/*.json*`, and this README.

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

To make the Vue dashboard move when the dialogue AI is working, write an observer-only activity event:

```powershell
.\.venv\Scripts\python.exe observer_ui\record_activity.py `
  --kind tool_start `
  --status running `
  --title "Reading formal docs" `
  --summary "Dialogue AI is checking governance before execution." `
  --from-actor "dialogue_ai" `
  --to-actor "observer_ui" `
  --channel "observer_event"
```

For Chinese text, prefer JSON input instead of command-line string arguments. This avoids Windows argv encoding drift:

```powershell
@'
{
  "kind": "tool_start",
  "status": "running",
  "title": "對話框 AI 正在讀取正式文件",
  "summary": "這筆事件只寫入 observer artifact，不改 formal runtime。",
  "from_actor": "對話框 AI",
  "to_actor": "Vue 觀測層",
  "channel": "observer_event"
}
'@ | .\.venv\Scripts\python.exe observer_ui\record_activity.py --json-stdin
```

To make a shell command produce start/end events automatically, wrap it with `scripts/observed_command.py`:

```powershell
.\.venv\Scripts\python.exe scripts\observed_command.py `
  --title "Run observer smoke check" `
  --summary "Vue should pulse on command start and completion." `
  --related-artifact "observer_ui/server.py" `
  -- .\.venv\Scripts\python.exe observer_ui\server.py --check
```

This writes only `outputs/observer_events/latest_meta_activity.json` and `outputs/observer_events/meta_activity.jsonl`. It never writes formal `latest_*_decision.json`.
The Activity view renders these fields as a live `FROM -> TO` route so it is clear who sent what to whom.

The main UI uses a workflow-console layout:

- `System Map` is one large project-wide canvas with 21 concrete production, decision, governance, sandbox, observer, and offline-learning nodes
- `Live signal layer` keeps the architecture fixed and animates only the latest `from_actor -> to_actor` observer route
- `Contract Map` shows the only formal cross-repository interfaces
- `Experiment Tree` separates formal, planned, sandbox, offline, and archived research directions
- `Runtime` shows only the current execution state
- left navigation separates architecture, runtime, activity, and artifacts
- nodes show only identity and state; detailed responsibility, evidence, and boundary rules live in the right inspector
- observer, teacher, learner, dialogue AI, and human feedback remain visibly outside the formal control chain

The live signal layer does not infer control authority from animation. Source and receiver nodes enlarge while a task is running and for 60 seconds after its result; one packet moves along only that observer route. Formal state still comes only from contracts, decisions, reports, and deployment review artifacts.

## Architecture View

The project architecture panel answers five questions in order:

1. What is the core goal and final output?
2. What is inside or outside the formal system boundary?
3. Which major module owns execution, evidence, or decisions?
4. Which subsystem produced the current status?
5. Is a connection production data, a formal artifact, a decision write-back, or a read-only sidecar?

Visual rules:

- thick blue line: production data flow
- amber solid line: formal artifact / contract flow
- violet dashed line: formal decision write-back
- green dotted boundary: observer / teacher / learner sidecar with no runtime control

Node status is derived from formal artifacts. A completed training event is not displayed as a completed project when the Unity Reference Gate remains blocked.

Stable architecture is not inferred from the latest run. The read-only backend exposes:

```text
GET /api/catalog
```

The catalog records systems, components, repositories, contracts, resources, and experiment groups. Runtime badges remain separate and continue to come from formal artifacts.

## Data Sources

- `outputs/agent_events/latest_*_complete.json`
- `outputs/agent_decisions/latest_*_decision.json`
- `outputs/reports/mcmc_run_inventory.summary.json`
- `outputs/observer_events/latest_meta_activity.json`
- `outputs/observer_events/meta_activity.jsonl`
- latest `outputs/experiments/**/deployment_review.json`
- `experimental/scaffold_gs_probe/latest_teacher_loop_status.json`
- `D:\agent_test\outputs\offline_learning\historical_plus_scaffold_report.json`
- `D:\agent_test\outputs\offline_learning\augmented_pytorch_baseline_report.json`

## Guardrail

V0 intentionally has no pass/fail buttons. Future UI actions must call a formal CLI and produce `outcome_feedback` or `deployment_review` artifacts before teacher / learner can absorb the result.

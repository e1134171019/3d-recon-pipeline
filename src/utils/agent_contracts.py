from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


def infer_outputs_root(project_root: Path, candidate_path: Path) -> Path:
    candidate = candidate_path.resolve()
    for node in [candidate] + list(candidate.parents):
        if node.name in {"SfM_models", "3DGS_models"}:
            return node.parent
    for node in [candidate] + list(candidate.parents):
        if node.name == "outputs":
            return node
    return project_root / "outputs"


def find_latest_step_file(root: Path, pattern: str, step_prefix: str) -> Path | None:
    candidates = list(root.rglob(pattern))
    if not candidates:
        return None

    def _step(path: Path) -> int:
        digits = "".join(ch for ch in path.stem.replace(step_prefix, "") if ch.isdigit())
        return int(digits) if digits else -1

    return max(candidates, key=_step)


def _normalize_payload_dict(data: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key, value in data.items():
        if isinstance(value, Path):
            normalized[key] = str(value.resolve())
        else:
            normalized[key] = value
    return normalized


def write_stage_contract(
    *,
    project_root: Path,
    run_root: Path,
    stage: str,
    status: str,
    artifacts: dict[str, Any],
    metrics: dict[str, Any],
    params: dict[str, Any] | None = None,
    summary: str = "",
    run_id: str | None = None,
) -> dict[str, str]:
    timestamp = datetime.now().isoformat()
    run_root = run_root.resolve()
    reports_dir = run_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    events_root = (project_root / "outputs" / "agent_events").resolve()
    events_root.mkdir(parents=True, exist_ok=True)

    if not run_id:
        run_id = f"{run_root.name}_{stage}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    payload = {
        "schema_version": 1,
        "timestamp": timestamp,
        "run_id": run_id,
        "run_root": str(run_root),
        "stage": stage,
        "status": status,
        "summary": summary,
        "artifacts": _normalize_payload_dict(artifacts),
        "metrics": metrics,
        "params": _normalize_payload_dict(params or {}),
    }

    local_contract = reports_dir / f"agent_{stage}.json"
    event_file = events_root / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{stage}_{status}.json"
    latest_file = events_root / f"latest_{stage}.json"

    text = json.dumps(payload, indent=2, ensure_ascii=False)
    local_contract.write_text(text, encoding="utf-8")
    event_file.write_text(text, encoding="utf-8")
    latest_file.write_text(text, encoding="utf-8")

    return {
        "local_contract": str(local_contract),
        "event_file": str(event_file),
        "latest_file": str(latest_file),
    }


def agent_decisions_root(project_root: Path) -> Path:
    root = (project_root / "outputs" / "agent_decisions").resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


REQUIRED_STAGE_CONTRACT_KEYS = ("schema_version", "timestamp", "run_id", "run_root", "stage", "status")
DICT_STAGE_CONTRACT_KEYS = ("artifacts", "metrics", "params")
JSON_ENCODINGS = ("utf-8", "utf-8-sig")


class StageContractValidationError(ValueError):
    """Raised when a production-to-decision contract has an invalid shape."""


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


def validate_stage_contract(payload: dict[str, Any], *, source: str = "") -> dict[str, Any]:
    """Validate the minimum schema shared with the decision layer."""
    if not isinstance(payload, dict):
        raise StageContractValidationError(f"contract must be a JSON object:{source}")

    missing = [key for key in REQUIRED_STAGE_CONTRACT_KEYS if key not in payload]
    if missing:
        raise StageContractValidationError(f"contract missing required keys {missing}:{source}")

    normalized = dict(payload)
    for key in DICT_STAGE_CONTRACT_KEYS:
        value = normalized.get(key, {})
        if value is None:
            value = {}
        if not isinstance(value, dict):
            raise StageContractValidationError(f"contract field '{key}' must be an object:{source}")
        normalized[key] = value

    for key in ("run_id", "run_root", "stage", "status"):
        if not str(normalized.get(key, "")).strip():
            raise StageContractValidationError(f"contract field '{key}' cannot be empty:{source}")

    return normalized


def read_stage_contract(path: Path) -> dict[str, Any]:
    """Read and validate a stage contract emitted by this production layer."""
    last_error: Exception | None = None
    for encoding in JSON_ENCODINGS:
        try:
            payload = json.loads(path.read_text(encoding=encoding))
            return validate_stage_contract(payload, source=str(path))
        except Exception as exc:
            last_error = exc
    raise last_error if last_error is not None else StageContractValidationError(f"contract parse failed:{path}")


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
    payload = validate_stage_contract(payload, source="write_stage_contract")

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


def _decision_filename_for_stage(stage: str) -> str | None:
    mapping = {
        "sfm_complete": "latest_sfm_decision.json",
        "train_complete": "latest_train_decision.json",
        "export_complete": "latest_export_decision.json",
    }
    return mapping.get(stage)


def trigger_decision_layer(
    *,
    project_root: Path,
    contract_path: Path | str,
) -> dict[str, Any]:
    contract_p = Path(contract_path).resolve()
    if not contract_p.exists():
        return {"status": "skipped", "reason": f"contract_missing:{contract_p}"}

    agent_runner = Path(r"D:\agent_test\run_phase0.py")
    if not agent_runner.exists():
        return {"status": "skipped", "reason": f"agent_runner_missing:{agent_runner}"}

    try:
        payload = read_stage_contract(contract_p)
    except Exception as exc:
        return {"status": "failed", "reason": f"contract_parse_failed:{exc}"}

    stage = str(payload.get("stage") or "")
    decision_name = _decision_filename_for_stage(stage)
    decision_path = agent_decisions_root(project_root) / decision_name if decision_name else None
    before_mtime = decision_path.stat().st_mtime if decision_path and decision_path.exists() else None

    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("PYTHONUTF8", "1")
    result = subprocess.run(
        [sys.executable, str(agent_runner), "--contract", str(contract_p)],
        cwd=str(agent_runner.parent),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
        env=env,
    )

    after_mtime = decision_path.stat().st_mtime if decision_path and decision_path.exists() else None
    decision_updated = (
        decision_path is not None
        and decision_path.exists()
        and (before_mtime is None or after_mtime is None or after_mtime > before_mtime)
    )

    summary: dict[str, Any] = {
        "status": "completed" if result.returncode == 0 and decision_updated else "warning",
        "returncode": result.returncode,
        "stage": stage,
        "contract_path": str(contract_p),
        "decision_path": str(decision_path) if decision_path is not None else "",
        "decision_updated": decision_updated,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }
    if result.returncode != 0:
        summary["status"] = "failed"
    elif decision_path is None:
        summary["status"] = "warning"
        summary["reason"] = f"unknown_stage:{stage}"
    elif not decision_updated:
        summary["status"] = "warning"
        summary["reason"] = f"decision_not_updated:{decision_path}"

    return summary

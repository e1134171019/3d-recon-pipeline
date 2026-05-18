from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from unittest import mock

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import export_ply_unity, train_3dgs


def main() -> int:
    project_root = PROJECT_ROOT
    tmp_root = project_root / "outputs" / "tmp_tests"
    tmp_root.mkdir(parents=True, exist_ok=True)
    workspace = Path(tempfile.mkdtemp(prefix="decision_hook_smoke_", dir=str(tmp_root)))

    summary: dict[str, object] = {"workspace": str(workspace)}

    train_run_root = workspace / "train_run"
    train_contract_paths = {
        "local_contract": str(train_run_root / "reports" / "agent_train_complete.json"),
        "event_file": str(train_run_root / "reports" / "train_complete.json"),
        "latest_file": str(train_run_root / "reports" / "latest_train_complete.json"),
    }
    with mock.patch.object(
        train_3dgs,
        "trigger_decision_layer",
        return_value={
            "status": "warning",
            "reason": "decision_not_updated:C:/tmp/latest_train_decision.json",
            "decision_path": "C:/tmp/latest_train_decision.json",
        },
    ), mock.patch.object(train_3dgs.console, "print"):
        train_3dgs._trigger_train_decision(project_root, train_run_root, train_contract_paths)

    train_audit = train_run_root / "reports" / "agent_train_complete_decision_hook.json"
    summary["train_warning"] = {
        "audit_exists": train_audit.exists(),
        "audit_path": str(train_audit),
        "payload": json.loads(train_audit.read_text(encoding="utf-8")),
    }

    export_run_root = workspace / "export_run"
    export_contract_paths = {
        "local_contract": str(export_run_root / "reports" / "agent_export_complete.json"),
        "event_file": str(export_run_root / "reports" / "export_complete.json"),
        "latest_file": str(export_run_root / "reports" / "latest_export_complete.json"),
    }
    with mock.patch.object(
        export_ply_unity,
        "trigger_decision_layer",
        return_value={
            "status": "failed",
            "returncode": 7,
            "stderr": "simulated decision failure",
            "stdout": "",
            "decision_path": "",
        },
    ), mock.patch("builtins.print"):
        export_ply_unity._trigger_export_decision(project_root, export_run_root, export_contract_paths)

    export_audit = export_run_root / "reports" / "agent_export_complete_decision_hook.json"
    summary["export_failed"] = {
        "audit_exists": export_audit.exists(),
        "audit_path": str(export_audit),
        "payload": json.loads(export_audit.read_text(encoding="utf-8")),
    }

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

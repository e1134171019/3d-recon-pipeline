from __future__ import annotations

import json
import os
import shutil
import unittest
import uuid
from contextlib import contextmanager
from pathlib import Path
from unittest import mock

from src.utils.agent_contracts import (
    StageContractValidationError,
    _resolve_agent_runner,
    read_stage_contract,
    validate_stage_contract,
    write_decision_hook_audit,
    write_stage_contract,
    trigger_decision_layer,
)

TEST_TMP_ROOT = Path(__file__).resolve().parents[1] / "outputs" / "tmp_tests"


@contextmanager
def temp_workspace():
    TEST_TMP_ROOT.mkdir(parents=True, exist_ok=True)
    path = TEST_TMP_ROOT / f"agent_contract_{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=False)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


class AgentContractTests(unittest.TestCase):
    def test_write_and_read_stage_contract_validates_minimum_schema(self):
        with temp_workspace() as tmp:
            root = Path(tmp)
            paths = write_stage_contract(
                project_root=root,
                run_root=root / "outputs" / "experiments" / "run1",
                stage="train_complete",
                status="completed",
                artifacts={"stats_json": root / "stats.json"},
                metrics={"lpips": 0.2},
                params=None,
                summary="ok",
                run_id="run1",
            )

            payload = read_stage_contract(Path(paths["latest_file"]))
            self.assertEqual(payload["stage"], "train_complete")
            self.assertEqual(payload["params"], {})
            self.assertTrue(payload["artifacts"]["stats_json"].endswith("stats.json"))

    def test_validate_stage_contract_rejects_missing_or_wrong_shape(self):
        with self.assertRaises(StageContractValidationError):
            validate_stage_contract({"stage": "train_complete"}, source="bad")

        with self.assertRaises(StageContractValidationError):
            validate_stage_contract(
                {
                    "schema_version": 1,
                    "timestamp": "2026-04-26T00:00:00",
                    "run_id": "run1",
                    "run_root": "root",
                    "stage": "train_complete",
                    "status": "completed",
                    "artifacts": [],
                },
                source="bad",
            )

    def test_validate_stage_contract_rejects_non_iso_timestamp(self):
        with self.assertRaises(StageContractValidationError):
            validate_stage_contract(
                {
                    "schema_version": 1,
                    "timestamp": 12345,
                    "run_id": "run1",
                    "run_root": "root",
                    "stage": "train_complete",
                    "status": "completed",
                    "artifacts": {},
                    "metrics": {},
                    "params": {},
                },
                source="bad",
            )

        with self.assertRaises(StageContractValidationError):
            validate_stage_contract(
                {
                    "schema_version": 1,
                    "timestamp": "not-a-date",
                    "run_id": "run1",
                    "run_root": "root",
                    "stage": "train_complete",
                    "status": "completed",
                    "artifacts": {},
                    "metrics": {},
                    "params": {},
                },
                source="bad",
            )

    def test_read_stage_contract_accepts_utf8_sig(self):
        with temp_workspace() as tmp:
            path = Path(tmp) / "contract.json"
            path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "timestamp": "2026-04-26T00:00:00",
                        "run_id": "run1",
                        "run_root": "root",
                        "stage": "export_complete",
                        "status": "completed",
                        "artifacts": {},
                        "metrics": {},
                        "params": {},
                    }
                ),
                encoding="utf-8-sig",
            )
            self.assertEqual(read_stage_contract(path)["stage"], "export_complete")

    def test_resolve_agent_runner_uses_env_overrides(self):
        with mock.patch.dict("os.environ", {"AGENT_TEST_RUNNER": r"C:\custom\runner.py"}, clear=False):
            self.assertEqual(_resolve_agent_runner(), Path(r"C:\custom\runner.py"))

        with mock.patch.dict("os.environ", {"AGENT_TEST_RUNNER": "", "AGENT_TEST_ROOT": r"C:\custom\agent"}, clear=False):
            self.assertEqual(_resolve_agent_runner(), Path(r"C:\custom\agent") / "run_phase0.py")

        with mock.patch.dict("os.environ", {"AGENT_TEST_RUNNER": "", "AGENT_TEST_ROOT": ""}, clear=False):
            self.assertIsNone(_resolve_agent_runner())

    def test_write_decision_hook_audit_persists_warning_payload(self):
        with temp_workspace() as tmp:
            run_root = Path(tmp) / "outputs" / "experiments" / "run1"
            audit_path = write_decision_hook_audit(
                run_root=run_root,
                stage="train_complete",
                decision_result={
                    "status": "warning",
                    "reason": "decision_not_updated",
                    "decision_path": str(run_root / "decision.json"),
                },
            )

            payload = json.loads(Path(audit_path).read_text(encoding="utf-8"))
            self.assertEqual(payload["stage"], "train_complete")
            self.assertEqual(payload["hook_status"], "warning")
            self.assertEqual(payload["decision_result"]["reason"], "decision_not_updated")

    def test_trigger_decision_layer_runs_real_phase0_contract_loop(self):
        agent_root = os.environ.get("AGENT_TEST_ROOT", "")
        if not agent_root:
            self.skipTest("AGENT_TEST_ROOT is not configured")
        agent_runner = Path(agent_root) / "run_phase0.py"
        if not agent_runner.exists():
            self.skipTest("decision-layer runner not available on this machine")

        with temp_workspace() as tmp:
            root = Path(tmp)
            run_root = root / "outputs" / "experiments" / "run1"
            reports_dir = run_root / "reports"
            reports_dir.mkdir(parents=True, exist_ok=True)
            (reports_dir / "pointcloud_validation_report.json").write_text(
                json.dumps(
                    {
                        "can_proceed_to_3dgs": True,
                        "images_count": 853,
                        "registered_images_count": 853,
                        "points3d_count": 315775,
                    }
                ),
                encoding="utf-8",
            )
            paths = write_stage_contract(
                project_root=root,
                run_root=run_root,
                stage="train_complete",
                status="completed",
                artifacts={"pointcloud_report": reports_dir / "pointcloud_validation_report.json"},
                metrics={"psnr": 27.0, "ssim": 0.89, "lpips": 0.23, "num_gs": 1000000},
                params={"train_mode": "mcmc", "iterations": 30000},
                summary="integration contract",
                run_id="integration_probe",
            )

            with mock.patch.dict("os.environ", {"AGENT_TEST_ROOT": agent_root}, clear=False):
                result = trigger_decision_layer(
                    project_root=root,
                    contract_path=paths["latest_file"],
                )

            self.assertEqual(result["returncode"], 0)
            self.assertEqual(result["status"], "completed")
            decision_path = Path(result["decision_path"])
            self.assertTrue(decision_path.exists())
            self.assertTrue(decision_path.resolve().is_relative_to((root / "outputs" / "agent_decisions").resolve()))
            self.assertFalse(
                decision_path.resolve().is_relative_to(
                    (Path(__file__).resolve().parents[1] / "outputs" / "agent_decisions").resolve()
                )
            )
            payload = json.loads(decision_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["source_stage"], "train_complete")
            self.assertEqual(payload["run_id"], "integration_probe")
            self.assertIn(payload["decision"], {"hold_export", "continue_train", "approve_export"})

    def test_trigger_decision_layer_rejects_tmp_contract_with_formal_project_root(self):
        agent_root = os.environ.get("AGENT_TEST_ROOT", "")
        if not agent_root:
            self.skipTest("AGENT_TEST_ROOT is not configured")
        agent_runner = Path(agent_root) / "run_phase0.py"
        if not agent_runner.exists():
            self.skipTest("decision-layer runner not available on this machine")

        with temp_workspace() as tmp:
            root = Path(tmp)
            run_root = root / "outputs" / "experiments" / "run1"
            paths = write_stage_contract(
                project_root=root,
                run_root=run_root,
                stage="train_complete",
                status="completed",
                artifacts={},
                metrics={"lpips": 0.23},
                params={"train_mode": "mcmc"},
                summary="tmp contract",
                run_id="tmp_contract_probe",
            )

            with mock.patch.dict("os.environ", {"AGENT_TEST_ROOT": agent_root}, clear=False), mock.patch(
                "src.utils.agent_contracts.subprocess.run"
            ) as run_mock:
                result = trigger_decision_layer(
                    project_root=Path(__file__).resolve().parents[1],
                    contract_path=paths["latest_file"],
                )

            run_mock.assert_not_called()
            self.assertEqual(result["status"], "failed")
            self.assertEqual(result["reason"], "test_contract_outside_events_root")
            self.assertFalse(result["decision_updated"])


if __name__ == "__main__":
    unittest.main()

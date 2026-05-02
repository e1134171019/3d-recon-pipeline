from __future__ import annotations

import json
import shutil
import unittest
import uuid
from contextlib import contextmanager
from pathlib import Path

from src.utils.agent_contracts import (
    StageContractValidationError,
    read_stage_contract,
    validate_stage_contract,
    write_stage_contract,
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


if __name__ == "__main__":
    unittest.main()

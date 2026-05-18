from __future__ import annotations

import io
import json
import runpy
import unittest
from contextlib import redirect_stdout
from pathlib import Path


class VerifyDecisionHookAuditScriptTests(unittest.TestCase):
    def test_script_emits_json_summary_with_train_and_export_audits(self):
        script_path = Path(__file__).resolve().parents[1] / "scripts" / "verify_decision_hook_audit.py"
        stdout = io.StringIO()

        with self.assertRaises(SystemExit) as raised, redirect_stdout(stdout):
            runpy.run_path(str(script_path), run_name="__main__")

        self.assertEqual(raised.exception.code, 0)
        payload = json.loads(stdout.getvalue())
        self.assertTrue(payload["train_warning"]["audit_exists"])
        self.assertTrue(payload["export_failed"]["audit_exists"])
        self.assertEqual(payload["train_warning"]["payload"]["hook_status"], "warning")
        self.assertEqual(payload["export_failed"]["payload"]["hook_status"], "failed")


if __name__ == "__main__":
    unittest.main()

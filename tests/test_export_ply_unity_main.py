import json
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from src import export_ply_unity
from _workspace_temp import workspace_tempdir


def _make_splats():
    return {
        "means": torch.tensor(
            [[0.0, 1.0, 2.0], [3.0, -4.0, 5.0], [6.0, 7.0, -8.0]],
            dtype=torch.float32,
        ),
        "scales": torch.zeros((3, 3), dtype=torch.float32),
        "quats": torch.tensor(
            [[1.0, 0.1, 0.2, 0.3], [1.0, 0.2, 0.3, 0.4], [1.0, 0.3, 0.4, 0.5]],
            dtype=torch.float32,
        ),
        "opacities": torch.tensor([-10.0, 0.0, 10.0], dtype=torch.float32),
        "sh0": torch.zeros((3, 1, 3), dtype=torch.float32),
        "shN": torch.zeros((3, 15, 3), dtype=torch.float32),
    }


def _save_checkpoint(path: Path):
    torch.save({"step": 12, "splats": _make_splats()}, path)


class ExportPlyUnityMainTests(unittest.TestCase):
    def _run_main(
        self,
        tmp: Path,
        *,
        extra_args=None,
        decision_result=None,
        data_dir=True,
        params_json=True,
    ):
        ckpt_path = tmp / "ckpt.pt"
        out_path = tmp / "outputs" / "3DGS_models" / "ply" / "unity.ply"
        data_path = tmp / "scene"
        params_path = tmp / "unity_params.json"
        _save_checkpoint(ckpt_path)
        if data_dir:
            data_path.mkdir(parents=True)
        if params_json:
            params_path.write_text(
                json.dumps(
                    {
                        "recommended_params": {
                            "min_opacity": 0.5,
                            "max_scale": 2.0,
                            "max_scale_percentile": 80.0,
                        }
                    }
                ),
                encoding="utf-8",
            )

        argv = [
            "export_ply_unity.py",
            "--ckpt",
            str(ckpt_path),
            "--data-dir",
            str(data_path),
            "--out",
            str(out_path),
            "--params-json",
            str(params_path),
        ]
        if extra_args:
            argv.extend(extra_args)

        contract_paths = {
            "local_contract": str(tmp / "agent_export_complete.json"),
            "event_file": str(tmp / "latest_export_complete.json"),
            "latest_file": str(tmp / "latest_export_complete.json"),
        }
        decision_result = decision_result or {
            "status": "completed",
            "decision_path": str(tmp / "latest_export_decision.json"),
        }

        with (
            mock.patch.object(sys, "argv", argv),
            mock.patch.object(
                export_ply_unity,
                "_reconstruct_normalize_transform",
                return_value=np.eye(4, dtype=np.float32),
            ) as reconstruct,
            mock.patch.object(export_ply_unity, "write_stage_contract", return_value=contract_paths) as write_contract,
            mock.patch.object(export_ply_unity, "trigger_decision_layer", return_value=decision_result) as trigger,
            mock.patch("builtins.print"),
        ):
            export_ply_unity.main()

        return out_path, reconstruct, write_contract, trigger

    def test_main_exports_with_params_denormalizes_filters_and_triggers_decision(self):
        with workspace_tempdir("export_unity_main_") as tmp:
            out_path, reconstruct, write_contract, trigger = self._run_main(
                tmp,
                extra_args=["--unity"],
            )

            self.assertTrue(out_path.exists())
            reconstruct.assert_called_once()
            kwargs = write_contract.call_args.kwargs
            self.assertEqual(kwargs["stage"], "export_complete")
            self.assertEqual(kwargs["status"], "completed")
            self.assertEqual(kwargs["metrics"]["num_splats"], 2)
            self.assertEqual(kwargs["metrics"]["min_opacity"], 0.5)
            self.assertEqual(kwargs["metrics"]["max_scale"], 2.0)
            self.assertEqual(kwargs["metrics"]["max_scale_percentile"], 80.0)
            self.assertTrue(kwargs["params"]["unity"])
            self.assertFalse(kwargs["params"]["no_denormalize"])
            trigger.assert_called_once()

    def test_main_accepts_no_denormalize_and_warning_decision_result(self):
        with workspace_tempdir("export_unity_warning_") as tmp:
            out_path, reconstruct, write_contract, trigger = self._run_main(
                tmp,
                extra_args=["--no-denormalize"],
                decision_result={"status": "warning", "reason": "decision_not_updated"},
                data_dir=False,
            )

            self.assertTrue(out_path.exists())
            reconstruct.assert_not_called()
            self.assertTrue(write_contract.call_args.kwargs["params"]["no_denormalize"])
            trigger.assert_called_once()

    def test_main_reports_failed_decision_result(self):
        with workspace_tempdir("export_unity_failed_decision_") as tmp:
            out_path, _, _, trigger = self._run_main(
                tmp,
                extra_args=["--no-denormalize"],
                decision_result={"status": "failed", "returncode": 2, "stderr": "boom"},
                data_dir=False,
            )

            self.assertTrue(out_path.exists())
            trigger.assert_called_once()

    def test_main_rejects_missing_checkpoint(self):
        with workspace_tempdir("export_unity_missing_ckpt_") as tmp:
            argv = ["export_ply_unity.py", "--ckpt", str(tmp / "missing.pt")]
            with mock.patch.object(sys, "argv", argv), mock.patch("builtins.print"):
                with self.assertRaises(SystemExit) as raised:
                    export_ply_unity.main()
            self.assertEqual(raised.exception.code, 1)

    def test_main_rejects_missing_params_json(self):
        with workspace_tempdir("export_unity_missing_params_") as tmp:
            ckpt_path = tmp / "ckpt.pt"
            _save_checkpoint(ckpt_path)
            argv = [
                "export_ply_unity.py",
                "--ckpt",
                str(ckpt_path),
                "--params-json",
                str(tmp / "missing.json"),
            ]
            with mock.patch.object(sys, "argv", argv), mock.patch("builtins.print"):
                with self.assertRaises(SystemExit) as raised:
                    export_ply_unity.main()
            self.assertEqual(raised.exception.code, 1)

    def test_main_rejects_missing_data_dir_when_denormalize_enabled(self):
        with workspace_tempdir("export_unity_missing_data_") as tmp:
            ckpt_path = tmp / "ckpt.pt"
            _save_checkpoint(ckpt_path)
            argv = [
                "export_ply_unity.py",
                "--ckpt",
                str(ckpt_path),
                "--data-dir",
                str(tmp / "missing_scene"),
                "--out",
                str(tmp / "out.ply"),
            ]
            with mock.patch.object(sys, "argv", argv), mock.patch("builtins.print"):
                with self.assertRaises(SystemExit) as raised:
                    export_ply_unity.main()
            self.assertEqual(raised.exception.code, 1)

    def test_reconstruct_normalize_transform_uses_colmap_parser(self):
        parser_calls = []

        class FakeParser:
            def __init__(self, **kwargs):
                parser_calls.append(kwargs)
                self.transform = np.eye(4, dtype=np.float32) * 2.0

        datasets_module = types.ModuleType("datasets")
        colmap_module = types.ModuleType("datasets.colmap")
        colmap_module.Parser = FakeParser

        with mock.patch.dict(sys.modules, {"datasets": datasets_module, "datasets.colmap": colmap_module}):
            transform = export_ply_unity._reconstruct_normalize_transform("scene-root")

        np.testing.assert_allclose(transform, np.eye(4, dtype=np.float32) * 2.0)
        self.assertEqual(
            parser_calls,
            [{"data_dir": "scene-root", "factor": 1, "normalize": True, "test_every": 8}],
        )

    def test_rotmat_to_quat_covers_all_shepperd_branches(self):
        rotations = [
            (np.diag([1.0, -1.0, -1.0]), np.array([0.0, 1.0, 0.0, 0.0])),
            (np.diag([-1.0, 1.0, -1.0]), np.array([0.0, 0.0, 1.0, 0.0])),
            (np.diag([-1.0, -1.0, 1.0]), np.array([0.0, 0.0, 0.0, 1.0])),
        ]
        for rotation, expected in rotations:
            quat = export_ply_unity.rotmat_to_quat(rotation)
            np.testing.assert_allclose(np.abs(quat), expected, atol=1e-6)

    def test_apply_export_filters_supports_percentile_threshold(self):
        means = np.array([[0.0, 0.0, 0.0]] * 3, dtype=np.float32)
        scales = np.log(
            np.array(
                [
                    [0.5, 0.5, 0.5],
                    [1.0, 1.0, 1.0],
                    [10.0, 10.0, 10.0],
                ],
                dtype=np.float32,
            )
        )
        quats = np.ones((3, 4), dtype=np.float32)
        opacities = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        sh0 = np.zeros((3, 1, 3), dtype=np.float32)
        shN = np.zeros((3, 15, 3), dtype=np.float32)

        with mock.patch("builtins.print"):
            out = export_ply_unity._apply_export_filters(
                means,
                scales,
                quats,
                opacities,
                sh0,
                shN,
                max_scale_percentile=80.0,
            )

        filtered_means, _, _, _, _, _, meta = out
        self.assertEqual(filtered_means.shape[0], 2)
        self.assertGreater(meta["effective_max_scale"], 1.0)
        self.assertLess(meta["effective_max_scale"], 10.0)

    def test_apply_export_filters_rejects_invalid_percentile(self):
        means = np.zeros((1, 3), dtype=np.float32)
        scales = np.zeros((1, 3), dtype=np.float32)
        quats = np.ones((1, 4), dtype=np.float32)
        opacities = np.zeros((1,), dtype=np.float32)
        sh0 = np.zeros((1, 1, 3), dtype=np.float32)
        shN = np.zeros((1, 15, 3), dtype=np.float32)

        with self.assertRaises(ValueError):
            export_ply_unity._apply_export_filters(
                means,
                scales,
                quats,
                opacities,
                sh0,
                shN,
                max_scale_percentile=120.0,
            )


if __name__ == "__main__":
    unittest.main()

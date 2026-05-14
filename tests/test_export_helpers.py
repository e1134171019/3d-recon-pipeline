import unittest
import sys
import types
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from src import export_ply, export_ply_unity
from _workspace_temp import workspace_tempdir


class ExportHelpersTests(unittest.TestCase):
    def _save_standard_checkpoint(self, path: Path):
        splats = {
            "means": torch.tensor([[0.0, 1.0, 2.0]], dtype=torch.float32),
            "scales": torch.tensor([[0.1, 0.2, 0.3]], dtype=torch.float32),
            "quats": torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32),
            "opacities": torch.tensor([0.5], dtype=torch.float32),
            "sh0": torch.zeros((1, 1, 3), dtype=torch.float32),
            "shN": torch.zeros((1, 15, 3), dtype=torch.float32),
        }
        torch.save({"step": 99, "splats": splats}, path)

    def test_rotmat_to_quat_identity(self):
        quat = export_ply_unity.rotmat_to_quat(np.eye(3, dtype=np.float64))
        np.testing.assert_allclose(quat, np.array([1.0, 0.0, 0.0, 0.0]), atol=1e-6)

    def test_quat_multiply_identity(self):
        identity = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        quats = np.array([[1.0, 0.0, 0.0, 0.0], [0.9238795, 0.3826834, 0.0, 0.0]])
        out = export_ply_unity.quat_multiply(identity, quats)
        np.testing.assert_allclose(out, quats, atol=1e-6)

    def test_denormalize_splats_identity_transform(self):
        means = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        quats = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        scales = np.array([[0.1, 0.2, 0.3]], dtype=np.float32)
        with mock.patch("builtins.print"):
            out_means, out_quats, out_scales = export_ply_unity.denormalize_splats(
                means, quats, scales, np.eye(4, dtype=np.float32)
            )
        np.testing.assert_allclose(out_means, means, atol=1e-6)
        np.testing.assert_allclose(out_quats, quats, atol=1e-6)
        np.testing.assert_allclose(out_scales, scales, atol=1e-6)

    def test_apply_unity_sh_handedness_flips_y_odd_bands_only(self):
        shN = np.arange(1, 1 + 15 * 3, dtype=np.float32).reshape(1, 15, 3)
        out = export_ply_unity._apply_unity_sh_handedness(shN)

        flipped = {0, 3, 4, 8, 9, 10}
        for idx in range(15):
            if idx in flipped:
                np.testing.assert_allclose(out[:, idx, :], -shN[:, idx, :])
            else:
                np.testing.assert_allclose(out[:, idx, :], shN[:, idx, :])

    def test_apply_unity_sh_handedness_truncates_safely(self):
        shN = np.ones((2, 4, 3), dtype=np.float32)
        out = export_ply_unity._apply_unity_sh_handedness(shN)
        np.testing.assert_allclose(out[:, 0, :], -1.0)
        np.testing.assert_allclose(out[:, 1:3, :], 1.0)
        np.testing.assert_allclose(out[:, 3, :], -1.0)

    def test_write_ply_unity_creates_binary_ply(self):
        with workspace_tempdir("export_unity_") as tmp:
            out_path = tmp / "unity.ply"
            means = np.array([[0.0, 1.0, 2.0]], dtype=np.float32)
            scales = np.array([[0.1, 0.2, 0.3]], dtype=np.float32)
            quats = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
            opacities = np.array([0.5], dtype=np.float32)
            sh0 = np.zeros((1, 1, 3), dtype=np.float32)
            shN = np.zeros((1, 15, 3), dtype=np.float32)

            with mock.patch("builtins.print"):
                export_ply_unity._write_ply(means, scales, quats, opacities, sh0, shN, out_path)

            header = out_path.read_bytes().split(b"end_header\n", 1)[0].decode("ascii")
            self.assertIn("format binary_little_endian 1.0", header)
            self.assertIn("element vertex 1", header)
            self.assertIn("property float rot_3", header)

    def test_write_ply_manual_creates_binary_ply(self):
        with workspace_tempdir("export_standard_") as tmp:
            out_path = tmp / "standard.ply"
            splats = {
                "means": torch.tensor([[0.0, 1.0, 2.0]], dtype=torch.float32),
                "scales": torch.tensor([[0.1, 0.2, 0.3]], dtype=torch.float32),
                "quats": torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32),
                "opacities": torch.tensor([0.5], dtype=torch.float32),
                "sh0": torch.zeros((1, 1, 3), dtype=torch.float32),
                "shN": torch.zeros((1, 15, 3), dtype=torch.float32),
            }

            with mock.patch("builtins.print"):
                export_ply._write_ply_manual(splats, out_path)

            header = out_path.read_bytes().split(b"end_header\n", 1)[0].decode("ascii")
            self.assertIn("format binary_little_endian 1.0", header)
            self.assertIn("element vertex 1", header)
            self.assertIn("property float opacity", header)

    def test_export_ply_main_rejects_missing_checkpoint(self):
        with workspace_tempdir("export_standard_missing_") as tmp:
            argv = ["export_ply.py", "--ckpt", str(tmp / "missing.pt")]
            with mock.patch.object(sys, "argv", argv), mock.patch("builtins.print"):
                with self.assertRaises(SystemExit) as raised:
                    export_ply.main()
            self.assertEqual(raised.exception.code, 1)

    def test_export_ply_main_uses_gsplat_export_when_available(self):
        with workspace_tempdir("export_standard_main_") as tmp:
            ckpt_path = tmp / "ckpt.pt"
            out_path = tmp / "point_cloud.ply"
            self._save_standard_checkpoint(ckpt_path)
            calls = []

            def fake_export_splats(**kwargs):
                calls.append(kwargs)
                Path(kwargs["save_to"]).write_bytes(b"ply\nend_header\n")

            gsplat_module = types.ModuleType("gsplat")
            utils_module = types.ModuleType("gsplat.utils")
            utils_module.export_splats = fake_export_splats

            argv = ["export_ply.py", "--ckpt", str(ckpt_path), "--out", str(out_path)]
            with (
                mock.patch.object(sys, "argv", argv),
                mock.patch.dict(sys.modules, {"gsplat": gsplat_module, "gsplat.utils": utils_module}),
                mock.patch("builtins.print"),
            ):
                export_ply.main()

            self.assertTrue(out_path.exists())
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0]["format"], "ply")
            self.assertEqual(calls[0]["save_to"], str(out_path))

    def test_export_ply_main_falls_back_to_manual_writer_on_import_error(self):
        with workspace_tempdir("export_standard_fallback_") as tmp:
            ckpt_path = tmp / "ckpt.pt"
            out_path = tmp / "point_cloud.ply"
            self._save_standard_checkpoint(ckpt_path)

            original_import = __import__

            def blocked_import(name, globals=None, locals=None, fromlist=(), level=0):
                if name == "gsplat.utils":
                    raise ImportError("export_splats missing")
                return original_import(name, globals, locals, fromlist, level)

            argv = ["export_ply.py", "--ckpt", str(ckpt_path), "--out", str(out_path)]
            with (
                mock.patch.object(sys, "argv", argv),
                mock.patch("builtins.__import__", side_effect=blocked_import),
                mock.patch("builtins.print"),
            ):
                export_ply.main()

            header = out_path.read_bytes().split(b"end_header\n", 1)[0].decode("ascii")
            self.assertIn("format binary_little_endian 1.0", header)
            self.assertIn("element vertex 1", header)


if __name__ == "__main__":
    unittest.main()

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from src import export_ply, export_ply_unity


class ExportHelpersTests(unittest.TestCase):
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

    def test_write_ply_unity_creates_binary_ply(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "unity.ply"
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
        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "standard.ply"
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


if __name__ == "__main__":
    unittest.main()

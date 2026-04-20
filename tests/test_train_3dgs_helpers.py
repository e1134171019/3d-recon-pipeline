import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src import train_3dgs


class Train3DGSHelpersTests(unittest.TestCase):
    def test_build_eval_schedule_adds_terminal_step(self):
        self.assertEqual(train_3dgs._build_eval_schedule(1000, 3000), [1000, 2000, 3000])
        self.assertEqual(train_3dgs._build_eval_schedule(0, 3000), [3000])
        self.assertEqual(train_3dgs._build_eval_schedule(2000, 3000), [2000, 3000])

    def test_load_train_params_supports_nested_payload(self):
        with tempfile.TemporaryDirectory() as tmp:
            params_path = Path(tmp) / "train_params.json"
            params_path.write_text(
                json.dumps(
                    {
                        "profile_name": "demo",
                        "train_params": {
                            "recommended_params": {
                                "iterations": 40000,
                                "grow_grad2d": 0.0008,
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            plan, recommended = train_3dgs._load_train_params(str(params_path))
            self.assertEqual(plan["recommended_params"]["iterations"], 40000)
            self.assertEqual(recommended["grow_grad2d"], 0.0008)

    def test_read_json_robust_supports_utf8_sig(self):
        with tempfile.TemporaryDirectory() as tmp:
            payload_path = Path(tmp) / "payload.json"
            payload_path.write_text(json.dumps({"ok": True}), encoding="utf-8-sig")
            parsed = train_3dgs._read_json_robust(payload_path)
            self.assertEqual(parsed, {"ok": True})

    def test_infer_reports_root_from_colmap_or_3dgs_dirs(self):
        project_root = Path(r"C:\project")
        colmap_root = project_root / "outputs" / "runs" / "run1" / "SfM_models" / "sift"
        train_root = project_root / "outputs" / "runs" / "run1" / "3DGS_models"
        self.assertEqual(
            train_3dgs._infer_reports_root(project_root, str(colmap_root)),
            project_root / "outputs" / "runs" / "run1" / "reports",
        )
        self.assertEqual(
            train_3dgs._infer_reports_root(project_root, str(train_root)),
            project_root / "outputs" / "runs" / "run1" / "reports",
        )

    def test_resolve_validation_report_prefers_inferred_reports_dir(self):
        project_root = Path(r"C:\project")
        path = train_3dgs._resolve_validation_report(
            project_root=project_root,
            colmap=r"outputs\runs\run1\SfM_models\sift\sparse\0",
            outdir=r"outputs\3DGS_models",
            validation_report="",
        )
        self.assertEqual(
            path,
            project_root / "outputs" / "runs" / "run1" / "reports" / "pointcloud_validation_report.json",
        )

    def test_resolve_sparse_model_dir_prefers_validation_report_sparse_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            project_root = Path(tmp)
            report_dir = project_root / "outputs" / "reports"
            report_dir.mkdir(parents=True)
            sparse_dir = project_root / "outputs" / "SfM_models" / "sift" / "sparse" / "2"
            sparse_dir.mkdir(parents=True)
            for name in ("cameras.bin", "images.bin", "points3D.bin"):
                (sparse_dir / name).write_bytes(b"x" * 32)

            validation_report = report_dir / "pointcloud_validation_report.json"
            validation_report.write_text(
                json.dumps({"sparse_dir": str(sparse_dir)}),
                encoding="utf-8",
            )

            with mock.patch.object(train_3dgs.console, "print"):
                resolved = train_3dgs._resolve_sparse_model_dir(
                    project_root=project_root,
                    colmap="outputs/SfM_models/sift/sparse/0",
                    validation_report_path=validation_report,
                )
            self.assertEqual(resolved, sparse_dir)

    def test_resolve_sparse_model_dir_picks_largest_available_sparse_child(self):
        with tempfile.TemporaryDirectory() as tmp:
            project_root = Path(tmp)
            sparse_root = project_root / "outputs" / "SfM_models" / "sift" / "sparse"
            sparse_root.mkdir(parents=True)
            candidate_a = sparse_root / "0"
            candidate_b = sparse_root / "1"
            candidate_a.mkdir()
            candidate_b.mkdir()

            for name, size_a, size_b in (
                ("cameras.bin", 64, 64),
                ("images.bin", 100, 500),
                ("points3D.bin", 200, 800),
            ):
                (candidate_a / name).write_bytes(b"a" * size_a)
                (candidate_b / name).write_bytes(b"b" * size_b)

            validation_report = project_root / "missing.json"
            with mock.patch.object(train_3dgs.console, "print"):
                resolved = train_3dgs._resolve_sparse_model_dir(
                    project_root=project_root,
                    colmap="outputs/SfM_models/sift",
                    validation_report_path=validation_report,
                )
            self.assertEqual(resolved, candidate_b)


if __name__ == "__main__":
    unittest.main()

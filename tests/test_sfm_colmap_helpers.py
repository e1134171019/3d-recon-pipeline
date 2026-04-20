import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src import sfm_colmap


def _build_feature_db(path: Path, keypoint_rows: list[int], descriptor_rows: list[int]) -> Path:
    conn = sqlite3.connect(path)
    cur = conn.cursor()
    cur.execute("CREATE TABLE keypoints(image_id INTEGER, rows INTEGER)")
    cur.execute("CREATE TABLE descriptors(image_id INTEGER, rows INTEGER)")
    cur.executemany("INSERT INTO keypoints VALUES (?, ?)", list(enumerate(keypoint_rows, start=1)))
    cur.executemany("INSERT INTO descriptors VALUES (?, ?)", list(enumerate(descriptor_rows, start=1)))
    conn.commit()
    conn.close()
    path.touch()
    path.write_bytes(path.read_bytes() + b"\0" * (11 * 1024 * 1024))
    return path


def _build_matching_db(path: Path, match_rows: list[int], inlier_rows: list[int]) -> Path:
    conn = sqlite3.connect(path)
    cur = conn.cursor()
    cur.execute("CREATE TABLE matches(pair_id INTEGER, rows INTEGER)")
    cur.execute("CREATE TABLE two_view_geometries(pair_id INTEGER, rows INTEGER)")
    cur.executemany("INSERT INTO matches VALUES (?, ?)", list(enumerate(match_rows, start=1)))
    cur.executemany("INSERT INTO two_view_geometries VALUES (?, ?)", list(enumerate(inlier_rows, start=1)))
    conn.commit()
    conn.close()
    path.touch()
    path.write_bytes(path.read_bytes() + b"\0" * (21 * 1024 * 1024))
    return path


class SfmColmapHelpersTests(unittest.TestCase):
    def test_find_glomap_prefers_local_installer_path(self):
        with mock.patch.object(Path, "exists", autospec=True) as mocked_exists:
            def side_effect(path_obj):
                return str(path_obj).endswith(r"installers\glomap-1.2.0\bin\glomap.exe")

            mocked_exists.side_effect = side_effect
            resolved = sfm_colmap.find_glomap(None)
        self.assertTrue(resolved.endswith(r"installers\glomap-1.2.0\bin\glomap.exe"))

    def test_run_mapper_step_uses_glomap_output_sparse_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            work_p = Path(tmp) / "SfM_models" / "sift"
            work_p.mkdir(parents=True)
            with mock.patch.object(sfm_colmap, "run") as mocked_run:
                sfm_colmap._run_mapper_step(
                    mapper_type="glomap",
                    colmap_exe="colmap.exe",
                    glomap_exe="glomap.exe",
                    db="database.db",
                    img=Path("images"),
                    work_p=work_p,
                )
            cmd = mocked_run.call_args[0][0]
            self.assertEqual(cmd[0:2], ["glomap.exe", "mapper"])
            self.assertIn(str((work_p / "sparse")), cmd)

    def test_infer_outputs_root(self):
        project_root = Path(r"C:\project")
        work_dir = project_root / "outputs" / "runs" / "r1" / "SfM_models" / "sift"
        inferred = sfm_colmap._infer_outputs_root(project_root, work_dir)
        self.assertEqual(inferred, project_root / "outputs" / "runs" / "r1")

    def test_find_best_sparse_model_prefers_more_registered_images_then_points(self):
        with tempfile.TemporaryDirectory() as tmp:
            sparse_root = Path(tmp)
            model0 = sparse_root / "0"
            model1 = sparse_root / "1"
            model0.mkdir()
            model1.mkdir()
            for folder in (model0, model1):
                for name in ("cameras.bin", "images.bin", "points3D.bin"):
                    (folder / name).write_bytes(b"x" * 128)

            with mock.patch.object(
                sfm_colmap,
                "_read_sparse_model_stats",
                side_effect=[
                    ({"registered_images_count": 10, "points3d_count": 100}, []),
                    ({"registered_images_count": 12, "points3d_count": 90}, []),
                ],
            ):
                best = sfm_colmap._find_best_sparse_model(sparse_root)
            self.assertEqual(best, model1)

    def test_find_best_sparse_model_accepts_nested_glomap_layout(self):
        with tempfile.TemporaryDirectory() as tmp:
            sparse_root = Path(tmp)
            nested = sparse_root / "0" / "0"
            nested.mkdir(parents=True)
            for name in ("cameras.bin", "images.bin", "points3D.bin"):
                (nested / name).write_bytes(b"x" * 128)

            with mock.patch.object(
                sfm_colmap,
                "_read_sparse_model_stats",
                return_value=(
                    {"registered_images_count": 20, "points3d_count": 500},
                    [],
                ),
            ):
                best = sfm_colmap._find_best_sparse_model(sparse_root)
            self.assertEqual(best, nested)

    def test_report_marks_pass_and_fail(self):
        with mock.patch("builtins.print"):
            ok = sfm_colmap._report("demo", {"count": 1}, [])
            fail = sfm_colmap._report("demo", {"count": 0}, ["bad"])
        self.assertTrue(ok["pass"])
        self.assertFalse(fail["pass"])
        self.assertEqual(fail["errors"], ["bad"])

    def test_get_matcher_time_estimate_and_mapper_assessment(self):
        self.assertIn("GPU", sfm_colmap._get_matcher_time_estimate(True))
        self.assertIn("CPU", sfm_colmap._get_matcher_time_estimate(False))
        assessment = sfm_colmap._assess_mapper_feasibility(20, 0.08, False)
        self.assertIn("較少", assessment)
        self.assertIn("偏低", assessment)

    def test_check_features_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = _build_feature_db(Path(tmp) / "database.db", [600, 800, 1000], [600, 800, 1000])
            with mock.patch("builtins.print"):
                result = sfm_colmap.check_features(str(db))
            self.assertTrue(result["pass"])
            self.assertEqual(result["cameras_count"], 3)
            self.assertEqual(result["total_features"], 2400)
            self.assertGreater(result["avg_features_per_image"], 500)

    def test_check_matching_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = _build_matching_db(Path(tmp) / "database.db", [500, 700, 800], [300, 500, 600])
            with mock.patch("builtins.print"):
                result = sfm_colmap.check_matching(str(db))
            self.assertTrue(result["pass"])
            self.assertEqual(result["num_matches"], 2000)
            self.assertEqual(result["num_inlier_matches"], 1400)
            self.assertGreater(result["inlier_ratio"], 0.1)

    def test_check_reconstruction_success_and_export_signals(self):
        with tempfile.TemporaryDirectory() as tmp:
            sparse_dir = Path(tmp) / "sparse" / "0"
            reports_dir = Path(tmp) / "reports"
            sparse_dir.mkdir(parents=True)
            reports_dir.mkdir()
            (sparse_dir / "cameras.bin").write_bytes(b"x" * 64)
            (sparse_dir / "images.bin").write_bytes(b"x" * 4096)
            (sparse_dir / "points3D.bin").write_bytes(b"x" * 16384)

            with mock.patch.object(
                sfm_colmap,
                "_read_sparse_model_stats",
                return_value=(
                    {
                        "cameras_count": 1,
                        "images_count": 60,
                        "registered_images_count": 60,
                        "points3d_count": 60000,
                    },
                    [],
                ),
            ):
                with mock.patch("builtins.print"):
                    result = sfm_colmap.check_reconstruction(str(sparse_dir))

            self.assertTrue(result["pass"])
            self.assertTrue(result["can_proceed_to_3dgs"])

            with mock.patch("builtins.print"):
                sfm_colmap.export_signals(result, str(sparse_dir), reports_dir)
            report_path = reports_dir / "pointcloud_validation_report.json"
            payload = None
            for encoding in ("utf-8", "utf-8-sig", "cp950"):
                try:
                    payload = json.loads(report_path.read_text(encoding=encoding))
                    break
                except Exception:
                    continue
            self.assertIsNotNone(payload)
            self.assertEqual(payload["points3d_count"], 60000)
            self.assertTrue(payload["can_proceed_to_3dgs"])


if __name__ == "__main__":
    unittest.main()

import json
import sqlite3
import subprocess
import unittest
import builtins
from pathlib import Path
from unittest import mock

from src import sfm_colmap
from _workspace_temp import workspace_tempdir


def _build_feature_db(path: Path, keypoint_rows: list[int], descriptor_rows: list[int]) -> Path:
    conn = sqlite3.connect(path)
    cur = conn.cursor()
    cur.execute("PRAGMA journal_mode=MEMORY")
    cur.execute("PRAGMA synchronous=OFF")
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
    cur.execute("PRAGMA journal_mode=MEMORY")
    cur.execute("PRAGMA synchronous=OFF")
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
    def test_run_echoes_and_delegates_to_subprocess(self):
        with mock.patch("builtins.print") as mocked_print, mock.patch.object(
            sfm_colmap.subprocess, "run"
        ) as mocked_run:
            sfm_colmap.run(["colmap", "help"])
        mocked_print.assert_called_once_with(">>", "colmap help")
        mocked_run.assert_called_once_with(["colmap", "help"], check=True)

    def test_find_colmap_resolves_argument_local_path_and_path_lookup(self):
        explicit = Path("C:/tools/colmap.exe")
        with mock.patch.object(Path, "exists", autospec=True, return_value=True):
            self.assertEqual(sfm_colmap.find_colmap(str(explicit)), str(explicit))

        with mock.patch.object(Path, "exists", autospec=True, return_value=False):
            with self.assertRaises(SystemExit):
                sfm_colmap.find_colmap("C:/missing/colmap.exe")

        def local_exists(path_obj):
            return str(path_obj).endswith(r"colmap\bin\colmap.exe")

        with mock.patch.object(Path, "exists", autospec=True, side_effect=local_exists):
            self.assertTrue(sfm_colmap.find_colmap(None).endswith(r"colmap\bin\colmap.exe"))

        with mock.patch.object(Path, "exists", autospec=True, return_value=False), mock.patch.object(
            sfm_colmap.shutil, "which", return_value="C:/PATH/colmap.exe"
        ):
            self.assertEqual(sfm_colmap.find_colmap(None), "C:/PATH/colmap.exe")

        with mock.patch.object(Path, "exists", autospec=True, return_value=False), mock.patch.object(
            sfm_colmap.shutil, "which", return_value=None
        ):
            with self.assertRaises(SystemExit):
                sfm_colmap.find_colmap(None)

    def test_find_glomap_prefers_local_installer_path(self):
        with mock.patch.object(Path, "exists", autospec=True) as mocked_exists:
            def side_effect(path_obj):
                return str(path_obj).endswith(r"installers\glomap-1.2.0\bin\glomap.exe")

            mocked_exists.side_effect = side_effect
            resolved = sfm_colmap.find_glomap(None)
        self.assertTrue(resolved.endswith(r"installers\glomap-1.2.0\bin\glomap.exe"))

    def test_find_glomap_resolves_argument_path_lookup_and_missing(self):
        explicit = Path("C:/tools/glomap.exe")
        with mock.patch.object(Path, "exists", autospec=True, return_value=True):
            self.assertEqual(sfm_colmap.find_glomap(str(explicit)), str(explicit))

        with mock.patch.object(Path, "exists", autospec=True, return_value=False):
            with self.assertRaises(SystemExit):
                sfm_colmap.find_glomap("C:/missing/glomap.exe")

        with mock.patch.object(Path, "exists", autospec=True, return_value=False), mock.patch.object(
            sfm_colmap.shutil, "which", return_value="C:/PATH/glomap.exe"
        ):
            self.assertEqual(sfm_colmap.find_glomap(None), "C:/PATH/glomap.exe")

        with mock.patch.object(Path, "exists", autospec=True, return_value=False), mock.patch.object(
            sfm_colmap.shutil, "which", return_value=None
        ):
            with self.assertRaises(SystemExit):
                sfm_colmap.find_glomap(None)

    def test_run_mapper_step_uses_glomap_output_sparse_root(self):
        with workspace_tempdir("sfm_mapper_") as tmp:
            work_p = tmp / "SfM_models" / "sift"
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

    def test_run_mapper_step_supports_incremental_and_rejects_bad_config(self):
        with workspace_tempdir("sfm_mapper_edges_") as tmp:
            work_p = tmp / "SfM_models" / "sift"
            work_p.mkdir(parents=True)
            with mock.patch.object(sfm_colmap, "run") as mocked_run:
                sfm_colmap._run_mapper_step(
                    mapper_type="incremental",
                    colmap_exe="colmap.exe",
                    glomap_exe=None,
                    db="database.db",
                    img=Path("images"),
                    work_p=work_p,
                )
            self.assertEqual(mocked_run.call_args.args[0][0:2], ["colmap.exe", "mapper"])

            with self.assertRaises(SystemExit):
                sfm_colmap._run_mapper_step(
                    mapper_type="glomap",
                    colmap_exe="colmap.exe",
                    glomap_exe=None,
                    db="database.db",
                    img=Path("images"),
                    work_p=work_p,
                )
            with self.assertRaises(SystemExit):
                sfm_colmap._run_mapper_step(
                    mapper_type="bad",
                    colmap_exe="colmap.exe",
                    glomap_exe=None,
                    db="database.db",
                    img=Path("images"),
                    work_p=work_p,
                )

    def test_infer_outputs_root(self):
        project_root = Path(r"C:\project")
        work_dir = project_root / "outputs" / "runs" / "r1" / "SfM_models" / "sift"
        inferred = sfm_colmap._infer_outputs_root(project_root, work_dir)
        self.assertEqual(inferred, project_root / "outputs" / "runs" / "r1")
        self.assertEqual(sfm_colmap._infer_outputs_root(project_root, project_root / "custom"), project_root / "outputs")

    def test_load_sfm_params_rejects_missing_and_bad_recommendations(self):
        with workspace_tempdir("sfm_params_edges_") as tmp:
            with self.assertRaises(SystemExit):
                sfm_colmap._load_sfm_params(str(tmp / "missing.json"))

            bad = tmp / "bad.json"
            bad.write_text(json.dumps({"recommended_params": []}), encoding="utf-8")
            with self.assertRaises(SystemExit):
                sfm_colmap._load_sfm_params(str(bad))

    def test_read_sparse_model_stats_uses_file_size_fallback_when_pycolmap_fails(self):
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "pycolmap":
                raise ImportError("missing pycolmap")
            return real_import(name, *args, **kwargs)

        with workspace_tempdir("sfm_sparse_stats_") as tmp, mock.patch(
            "builtins.__import__", side_effect=fake_import
        ):
            sparse = tmp / "0"
            sparse.mkdir()
            (sparse / "cameras.bin").write_bytes(b"x" * 128)
            (sparse / "images.bin").write_bytes(b"x" * 380000)
            (sparse / "points3D.bin").write_bytes(b"x" * 1480)

            stats, warnings = sfm_colmap._read_sparse_model_stats(sparse)

        self.assertEqual(stats["cameras_count"], 2)
        self.assertEqual(stats["images_count"], 2)
        self.assertEqual(stats["points3d_count"], 10)
        self.assertTrue(warnings)

    def test_find_best_sparse_model_prefers_more_registered_images_then_points(self):
        with workspace_tempdir("sfm_sparse_best_") as tmp:
            sparse_root = tmp
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
        with workspace_tempdir("sfm_sparse_nested_") as tmp:
            sparse_root = tmp
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

    def test_find_best_sparse_model_returns_none_for_missing_or_invalid_sparse_root(self):
        with workspace_tempdir("sfm_sparse_none_") as tmp:
            self.assertIsNone(sfm_colmap._find_best_sparse_model(tmp / "missing"))

            sparse_root = tmp / "sparse"
            sparse_root.mkdir()
            (sparse_root / "not_a_dir").write_text("skip", encoding="utf-8")
            invalid = sparse_root / "0"
            invalid.mkdir()
            (invalid / "nested_file").write_text("skip", encoding="utf-8")
            nested_invalid = invalid / "nested"
            nested_invalid.mkdir()
            (nested_invalid / "cameras.bin").write_bytes(b"x")

            self.assertIsNone(sfm_colmap._find_best_sparse_model(sparse_root))

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
        with workspace_tempdir("sfm_features_") as tmp:
            db = _build_feature_db(tmp / "database.db", [600, 800, 1000], [600, 800, 1000])
            with mock.patch("builtins.print"):
                result = sfm_colmap.check_features(str(db))
            self.assertTrue(result["pass"])
            self.assertEqual(result["cameras_count"], 3)
            self.assertEqual(result["total_features"], 2400)
            self.assertGreater(result["avg_features_per_image"], 500)

    def test_check_matching_success(self):
        with workspace_tempdir("sfm_matching_") as tmp:
            db = _build_matching_db(tmp / "database.db", [500, 700, 800], [300, 500, 600])
            with mock.patch("builtins.print"):
                result = sfm_colmap.check_matching(str(db))
            self.assertTrue(result["pass"])
            self.assertEqual(result["num_matches"], 2000)
            self.assertEqual(result["num_inlier_matches"], 1400)
            self.assertGreater(result["inlier_ratio"], 0.1)

    def test_check_reconstruction_success_and_export_signals(self):
        with workspace_tempdir("sfm_recon_") as tmp:
            sparse_dir = tmp / "sparse" / "0"
            reports_dir = tmp / "reports"
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

    def test_main_applies_agent_params_and_writes_contract(self):
        with workspace_tempdir("sfm_main_success_") as tmp:
            img = tmp / "images"
            work = tmp / "work"
            img.mkdir()
            work.mkdir()
            (img / "0001.jpg").write_bytes(b"image")
            params = tmp / "sfm_params.json"
            params.write_text(
                json.dumps(
                    {
                        "sfm_params": {
                            "profile_name": "coverage_probe",
                            "recommended_params": {
                                "imgdir": str(img),
                                "work": str(work),
                                "use_gpu": False,
                                "max_image_size": 1200,
                                "max_features": 9000,
                                "seq_overlap": 4,
                                "loop_detection": False,
                                "mapper_type": "glomap",
                                "sift_peak_threshold": 0.01,
                                "sift_edge_threshold": 12,
                                "colmap_bin": "C:/tools/colmap.exe",
                                "glomap_bin": "C:/tools/glomap.exe",
                                "resume": False,
                                "min_points3d": 1234,
                            },
                        }
                    }
                ),
                encoding="utf-8",
            )
            best_sparse = work / "sparse" / "1"

            with mock.patch.object(sfm_colmap, "find_colmap", return_value="colmap.exe") as find_colmap:
                with mock.patch.object(sfm_colmap, "find_glomap", return_value="glomap.exe") as find_glomap:
                    with mock.patch.object(sfm_colmap, "run") as mocked_run:
                        with mock.patch.object(
                            sfm_colmap,
                            "check_features",
                            return_value={
                                "pass": True,
                                "cameras_count": 40,
                                "total_features": 32000,
                            },
                        ):
                            with mock.patch.object(
                                sfm_colmap,
                                "check_matching",
                                return_value={
                                    "pass": True,
                                    "num_matches": 5000,
                                    "inlier_ratio": 0.25,
                                },
                            ):
                                with mock.patch.object(sfm_colmap, "_run_mapper_step") as mapper:
                                    with mock.patch.object(sfm_colmap, "_find_best_sparse_model", return_value=best_sparse):
                                        with mock.patch.object(
                                            sfm_colmap,
                                            "check_reconstruction",
                                            return_value={
                                                "pass": True,
                                                "cameras_count": 3,
                                                "images_count": 40,
                                                "registered_images_count": 40,
                                                "points3d_count": 70000,
                                                "can_proceed_to_3dgs": True,
                                            },
                                        ) as check_recon:
                                            with mock.patch.object(sfm_colmap, "export_signals") as export_signals:
                                                with mock.patch.object(
                                                    sfm_colmap,
                                                    "write_stage_contract",
                                                    return_value={
                                                        "local_contract": "local.json",
                                                        "event_file": "event.json",
                                                    },
                                                ) as write_contract:
                                                    with mock.patch.object(sfm_colmap.time, "time", side_effect=[100.0, 250.0]):
                                                        with mock.patch("builtins.print"):
                                                            sfm_colmap.main(params_json=str(params))

            find_colmap.assert_called_once_with("C:/tools/colmap.exe")
            find_glomap.assert_called_once_with("C:/tools/glomap.exe")
            self.assertEqual(mocked_run.call_count, 2)
            mapper_kwargs = mapper.call_args.kwargs
            self.assertEqual(mapper_kwargs["mapper_type"], "glomap")
            self.assertEqual(mapper_kwargs["glomap_exe"], "glomap.exe")
            check_recon.assert_called_once_with(str(best_sparse), min_points3d=1234)
            export_signals.assert_called_once()
            contract_kwargs = write_contract.call_args.kwargs
            self.assertEqual(contract_kwargs["stage"], "sfm_complete")
            self.assertEqual(contract_kwargs["status"], "completed")
            self.assertEqual(contract_kwargs["params"]["mapper_type"], "glomap")
            self.assertFalse(contract_kwargs["params"]["use_gpu"])

    def test_main_rejects_empty_image_directory(self):
        with workspace_tempdir("sfm_main_empty_") as tmp:
            img = tmp / "empty_images"
            work = tmp / "work"
            img.mkdir()
            with self.assertRaises(SystemExit) as ctx:
                sfm_colmap.main(imgdir=str(img), work=str(work))
            self.assertIn("影像夾為空", str(ctx.exception))

    def test_main_stops_when_feature_gate_fails(self):
        with workspace_tempdir("sfm_main_feature_fail_") as tmp:
            img = tmp / "images"
            work = tmp / "work"
            img.mkdir()
            (img / "0001.jpg").write_bytes(b"image")

            with mock.patch.object(sfm_colmap, "find_colmap", return_value="colmap.exe"):
                with mock.patch.object(sfm_colmap, "run"):
                    with mock.patch.object(sfm_colmap, "check_features", return_value={"pass": False}):
                        with self.assertRaises(SystemExit) as ctx:
                            sfm_colmap.main(imgdir=str(img), work=str(work))
            self.assertIn("Step 1 特徵提取驗證失敗", str(ctx.exception))

    def test_main_stops_when_matching_gate_fails(self):
        with workspace_tempdir("sfm_main_matching_fail_") as tmp:
            img = tmp / "images"
            work = tmp / "work"
            img.mkdir()
            (img / "0001.jpg").write_bytes(b"image")

            with mock.patch.object(sfm_colmap, "find_colmap", return_value="colmap.exe"):
                with mock.patch.object(sfm_colmap, "run"):
                    with mock.patch.object(
                        sfm_colmap,
                        "check_features",
                        return_value={"pass": True, "cameras_count": 10, "total_features": 8000},
                    ):
                        with mock.patch.object(
                            sfm_colmap,
                            "check_matching",
                            return_value={"pass": False, "inlier_ratio": 0.01},
                        ):
                            with mock.patch("builtins.print"):
                                with self.assertRaises(SystemExit) as ctx:
                                    sfm_colmap.main(imgdir=str(img), work=str(work))
            self.assertIn("Step 2 匹配驗證失敗", str(ctx.exception))

    def test_main_stops_when_mapper_outputs_no_sparse_model(self):
        with workspace_tempdir("sfm_main_no_sparse_") as tmp:
            img = tmp / "images"
            work = tmp / "work"
            img.mkdir()
            (img / "0001.jpg").write_bytes(b"image")

            with mock.patch.object(sfm_colmap, "find_colmap", return_value="colmap.exe"):
                with mock.patch.object(sfm_colmap, "run"):
                    with mock.patch.object(
                        sfm_colmap,
                        "check_features",
                        return_value={"pass": True, "cameras_count": 50, "total_features": 40000},
                    ):
                        with mock.patch.object(
                            sfm_colmap,
                            "check_matching",
                            return_value={"pass": True, "num_matches": 2000, "inlier_ratio": 0.2},
                        ):
                            with mock.patch.object(sfm_colmap, "_run_mapper_step"):
                                with mock.patch.object(sfm_colmap, "_find_best_sparse_model", return_value=None):
                                    with mock.patch("builtins.print"):
                                        with self.assertRaises(SystemExit) as ctx:
                                            sfm_colmap.main(imgdir=str(img), work=str(work))
            self.assertIn("COLMAP Mapper 失敗診斷", str(ctx.exception))

    def test_main_stops_when_reconstruction_gate_fails(self):
        with workspace_tempdir("sfm_main_recon_fail_") as tmp:
            img = tmp / "images"
            work = tmp / "work"
            img.mkdir()
            (img / "0001.jpg").write_bytes(b"image")
            best_sparse = work / "sparse" / "0"

            with mock.patch.object(sfm_colmap, "find_colmap", return_value="colmap.exe"):
                with mock.patch.object(sfm_colmap, "run"):
                    with mock.patch.object(
                        sfm_colmap,
                        "check_features",
                        return_value={"pass": True, "cameras_count": 50, "total_features": 40000},
                    ):
                        with mock.patch.object(
                            sfm_colmap,
                            "check_matching",
                            return_value={"pass": True, "num_matches": 2000, "inlier_ratio": 0.2},
                        ):
                            with mock.patch.object(sfm_colmap, "_run_mapper_step"):
                                with mock.patch.object(sfm_colmap, "_find_best_sparse_model", return_value=best_sparse):
                                    with mock.patch.object(
                                        sfm_colmap,
                                        "check_reconstruction",
                                        return_value={
                                            "pass": False,
                                            "errors": ["too sparse"],
                                        },
                                    ):
                                        with mock.patch("builtins.print"):
                                            with self.assertRaises(SystemExit) as ctx:
                                                sfm_colmap.main(imgdir=str(img), work=str(work))
            self.assertIn("Step 3 重建驗證失敗", str(ctx.exception))

    def test_main_wraps_colmap_subprocess_failure(self):
        with workspace_tempdir("sfm_main_subprocess_fail_") as tmp:
            img = tmp / "images"
            work = tmp / "work"
            img.mkdir()
            (img / "0001.jpg").write_bytes(b"image")

            with mock.patch.object(sfm_colmap, "find_colmap", return_value="colmap.exe"):
                with mock.patch.object(
                    sfm_colmap,
                    "run",
                    side_effect=subprocess.CalledProcessError(2, ["colmap"]),
                ):
                    with self.assertRaises(SystemExit) as ctx:
                        sfm_colmap.main(imgdir=str(img), work=str(work))
            self.assertIn("COLMAP 指令執行失敗", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()

import json
import unittest
from pathlib import Path
from unittest import mock

from src import train_3dgs
from _workspace_temp import workspace_tempdir


class Train3DGSHelpersTests(unittest.TestCase):
    def test_run_clears_stale_arch_list_before_subprocess(self):
        with mock.patch.dict(train_3dgs.os.environ, {"TORCH_CUDA_ARCH_LIST": "12.0"}, clear=False):
            with mock.patch.object(train_3dgs.subprocess, "run") as mocked_run, mock.patch.object(
                train_3dgs.console, "print"
            ) as mocked_print:
                train_3dgs._run(["python", "demo.py"], cwd="C:/repo")

        kwargs = mocked_run.call_args.kwargs
        self.assertEqual(kwargs["cwd"], "C:/repo")
        self.assertNotIn("TORCH_CUDA_ARCH_LIST", kwargs["env"])
        self.assertEqual(kwargs["env"]["PYTHONIOENCODING"], "utf-8")
        self.assertEqual(kwargs["env"]["PYTHONUTF8"], "1")
        mocked_print.assert_called()

    def test_build_eval_schedule_adds_terminal_step(self):
        self.assertEqual(train_3dgs._build_eval_schedule(1000, 3000), [1000, 2000, 3000])
        self.assertEqual(train_3dgs._build_eval_schedule(0, 3000), [3000])
        self.assertEqual(train_3dgs._build_eval_schedule(2000, 3000), [2000, 3000])

    def test_load_train_params_supports_nested_payload(self):
        with workspace_tempdir("train_params_") as tmp:
            params_path = tmp / "train_params.json"
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
        with workspace_tempdir("train_json_") as tmp:
            payload_path = tmp / "payload.json"
            payload_path.write_text(json.dumps({"ok": True}), encoding="utf-8-sig")
            parsed = train_3dgs._read_json_robust(payload_path)
            self.assertEqual(parsed, {"ok": True})

    def test_resolve_imgdir_falls_back_to_frames_cleaned_for_default_path(self):
        with workspace_tempdir("train_imgdir_") as tmp:
            project_root = tmp
            fallback = project_root / "data" / "frames_cleaned"
            fallback.mkdir(parents=True)
            (fallback / "frame_000001.jpg").write_bytes(b"jpg")

            with mock.patch.object(train_3dgs.console, "print") as mocked_print:
                resolved = train_3dgs._resolve_imgdir(project_root, "data/frames_1600")

            self.assertEqual(resolved, fallback)
            mocked_print.assert_called()

    def test_resolve_imgdir_returns_requested_when_present(self):
        with workspace_tempdir("train_imgdir_present_") as tmp:
            project_root = tmp
            requested = project_root / "custom_frames"
            requested.mkdir(parents=True)
            (requested / "frame_000001.png").write_bytes(b"png")

            resolved = train_3dgs._resolve_imgdir(project_root, str(requested))
            self.assertEqual(resolved, requested)

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

    def test_check_pointcloud_validation_missing_report_raises_file_not_found(self):
        with workspace_tempdir("train_report_missing_") as tmp, mock.patch.object(
            train_3dgs.console, "print"
        ):
            missing = tmp / "missing.json"
            with self.assertRaises(FileNotFoundError):
                train_3dgs._check_pointcloud_validation(str(missing))

    def test_check_pointcloud_validation_invalid_report_exits(self):
        with workspace_tempdir("train_report_invalid_") as tmp, mock.patch.object(
            train_3dgs.console, "print"
        ):
            report = tmp / "bad.json"
            report.write_text("{not-json", encoding="utf-8")
            with self.assertRaises(train_3dgs.typer.Exit):
                train_3dgs._check_pointcloud_validation(str(report))

    def test_check_pointcloud_validation_failed_report_exits(self):
        with workspace_tempdir("train_report_fail_") as tmp, mock.patch.object(
            train_3dgs.console, "print"
        ):
            report = tmp / "report.json"
            report.write_text(
                json.dumps({"can_proceed_to_3dgs": False, "diagnosis": "too sparse"}),
                encoding="utf-8",
            )
            with self.assertRaises(train_3dgs.typer.Exit):
                train_3dgs._check_pointcloud_validation(str(report))

    def test_check_pointcloud_validation_passes_when_gate_is_open(self):
        with workspace_tempdir("train_report_pass_") as tmp, mock.patch.object(
            train_3dgs.console, "print"
        ):
            report = tmp / "report.json"
            report.write_text(json.dumps({"can_proceed_to_3dgs": True}), encoding="utf-8")
            self.assertTrue(train_3dgs._check_pointcloud_validation(str(report)))

    def test_resolve_sparse_model_dir_prefers_validation_report_sparse_dir(self):
        with workspace_tempdir("train_sparse_report_") as tmp:
            project_root = tmp
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
        with workspace_tempdir("train_sparse_best_") as tmp:
            project_root = tmp
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

    def test_remove_path_deletes_file_and_directory(self):
        file_path = mock.MagicMock()
        file_path.exists.return_value = True
        file_path.is_symlink.return_value = False
        file_path.is_file.return_value = True

        dir_path = mock.MagicMock()
        dir_path.exists.return_value = True
        dir_path.is_symlink.return_value = False
        dir_path.is_file.return_value = False

        with mock.patch.object(train_3dgs.os, "rmdir", side_effect=OSError), mock.patch.object(
            train_3dgs.shutil, "rmtree"
        ) as mocked_rmtree:
            train_3dgs._remove_path(file_path)
            train_3dgs._remove_path(dir_path)

        file_path.unlink.assert_called_once()
        mocked_rmtree.assert_called_once_with(dir_path)

    def test_ensure_scene_dir_rebuilds_links_from_scratch(self):
        with workspace_tempdir("train_scene_dir_") as tmp:
            scene_dir = tmp / "scene"
            img_target = tmp / "images_src"
            sparse_target = tmp / "sparse_src"
            img_target.mkdir()
            sparse_target.mkdir()

            with mock.patch.object(train_3dgs, "_remove_path") as mocked_remove, mock.patch.object(
                train_3dgs, "_create_directory_link"
            ) as mocked_link:
                train_3dgs._ensure_scene_dir(scene_dir, img_target, sparse_target)

            self.assertTrue((scene_dir / "sparse").exists())
            mocked_remove.assert_not_called()
            self.assertEqual(mocked_link.call_count, 2)
            mocked_link.assert_any_call(scene_dir / "images", img_target)
            mocked_link.assert_any_call(scene_dir / "sparse" / "0", sparse_target)

    def test_resolve_effective_train_config_uses_mcmc_preset_defaults(self):
        effective = train_3dgs._resolve_effective_train_config(
            "mcmc",
            absgrad=False,
            grow_grad2d=0.0002,
            antialiased=False,
            random_bkgd=False,
            cap_max=1_000_000,
            mcmc_min_opacity=None,
            mcmc_noise_lr=None,
            opacity_reg=0.0,
            pose_opt=False,
            app_opt=False,
        )
        self.assertEqual(effective["opacity_reg"], 0.01)
        self.assertEqual(effective["mcmc_min_opacity"], 0.005)
        self.assertEqual(effective["mcmc_noise_lr"], 500000.0)
        self.assertEqual(effective["mcmc_scale_reg"], 0.01)

    def test_resolve_effective_train_config_preserves_explicit_mcmc_override(self):
        effective = train_3dgs._resolve_effective_train_config(
            "mcmc",
            absgrad=False,
            grow_grad2d=0.0002,
            antialiased=True,
            random_bkgd=True,
            cap_max=750000,
            mcmc_min_opacity=0.01,
            mcmc_noise_lr=100000.0,
            opacity_reg=0.02,
            pose_opt=True,
            app_opt=False,
        )
        self.assertEqual(effective["opacity_reg"], 0.02)
        self.assertEqual(effective["cap_max"], 750000)
        self.assertEqual(effective["mcmc_min_opacity"], 0.01)
        self.assertEqual(effective["mcmc_noise_lr"], 100000.0)
        self.assertTrue(effective["antialiased"])
        self.assertTrue(effective["random_bkgd"])
        self.assertTrue(effective["pose_opt"])

    def test_main_builds_mcmc_command_with_effective_overrides(self):
        with workspace_tempdir("train_main_") as tmp:
            imgdir = tmp / "images"
            colmap = tmp / "sparse" / "0"
            outdir = tmp / "out"
            report = tmp / "pointcloud_validation_report.json"
            imgdir.mkdir(parents=True)
            colmap.mkdir(parents=True)
            (imgdir / "frame_000001.jpg").write_bytes(b"jpg")
            report.write_text(json.dumps({"can_proceed_to_3dgs": True}), encoding="utf-8")

            with mock.patch.object(train_3dgs, "_check_gsplat", return_value=True), mock.patch.object(
                train_3dgs, "_check_pointcloud_validation", return_value=True
            ), mock.patch.object(train_3dgs, "_ensure_scene_dir") as mocked_scene_dir, mock.patch.object(
                train_3dgs, "_run"
            ) as mocked_run, mock.patch.object(
                train_3dgs, "find_latest_step_file", side_effect=[None, None]
            ), mock.patch.object(
                train_3dgs, "infer_outputs_root", return_value=tmp / "outputs_root"
            ), mock.patch.object(
                train_3dgs,
                "write_stage_contract",
                return_value={
                    "local_contract": tmp / "agent_train_complete.json",
                    "event_file": tmp / "latest_train_complete.json",
                },
            ) as mocked_contract, mock.patch.object(train_3dgs.console, "print"):
                train_3dgs.main(
                    train_mode="mcmc",
                    imgdir=str(imgdir),
                    colmap=str(colmap),
                    outdir=str(outdir),
                    iterations=7000,
                    sh_degree=3,
                    densify_until=15000,
                    scene_scale=0.0,
                    scale_json="",
                    eval_steps=1000,
                    data_factor=1,
                    absgrad=False,
                    grow_grad2d=0.0002,
                    antialiased=True,
                    random_bkgd=True,
                    cap_max=750000,
                    mcmc_min_opacity=0.01,
                    mcmc_noise_lr=100000.0,
                    opacity_reg=0.0,
                    pose_opt=False,
                    app_opt=False,
                    disable_video=True,
                    loss_mask_dir="",
                    validation_report=str(report),
                    params_json="",
                )

            mocked_scene_dir.assert_called_once()
            cmd = mocked_run.call_args.args[0]
            cwd = mocked_run.call_args.kwargs["cwd"]
            self.assertTrue(str(cmd[0]).endswith("python.exe"))
            self.assertIn("mcmc", cmd)
            self.assertIn("--strategy.cap-max", cmd)
            self.assertIn("750000", cmd)
            self.assertIn("--strategy.min-opacity", cmd)
            self.assertIn("0.01", cmd)
            self.assertIn("--strategy.noise-lr", cmd)
            self.assertIn("100000.0", cmd)
            self.assertIn("--antialiased", cmd)
            self.assertIn("--random-bkgd", cmd)
            self.assertIn("--disable-video", cmd)
            self.assertEqual(cwd, str((Path(train_3dgs.__file__).parent.parent / "gsplat_runner").resolve()))
            mocked_contract.assert_called_once()


if __name__ == "__main__":
    unittest.main()

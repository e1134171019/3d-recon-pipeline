import unittest
import os
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

from src import downscale_frames, preprocess_phase0
from _workspace_temp import workspace_tempdir


class DummyProgress:
    def __init__(self, *_, **__):
        self.updated = 0

    def update(self, value):
        self.updated += value

    def close(self):
        pass


class FakeVideoCapture:
    def __init__(self, frames=None, *, opened=True, fps=2.0):
        self.frames = list(frames or [])
        self.opened = opened
        self.fps = fps
        self.index = 0
        self.released = False

    def isOpened(self):
        return self.opened

    def get(self, prop):
        if prop == cv2.CAP_PROP_FRAME_COUNT:
            return len(self.frames)
        if prop == cv2.CAP_PROP_FPS:
            return self.fps
        return 0

    def read(self):
        if self.index >= len(self.frames):
            return False, None
        frame = self.frames[self.index]
        self.index += 1
        return True, frame.copy()

    def release(self):
        self.released = True


class PreprocessAndDownscaleTests(unittest.TestCase):
    def test_get_video_path_handles_missing_empty_and_multiple_files(self):
        with workspace_tempdir("preprocess_video_path_") as tmp:
            previous_cwd = os.getcwd()
            os.chdir(tmp)
            try:
                with mock.patch("builtins.print"):
                    self.assertIsNone(preprocess_phase0.get_video_path())

                video_dir = tmp / "data" / "viode"
                video_dir.mkdir(parents=True)
                with mock.patch("builtins.print"):
                    self.assertIsNone(preprocess_phase0.get_video_path())

                first = video_dir / "sample.mp4"
                second = video_dir / "sample.avi"
                first.write_bytes(b"mp4")
                second.write_bytes(b"avi")
                with mock.patch("builtins.print"):
                    selected = preprocess_phase0.get_video_path()
                self.assertEqual(selected, Path("data/viode/sample.mp4"))
            finally:
                os.chdir(previous_cwd)

    def test_gamma_and_highlight_helpers_change_pixels(self):
        frame = np.full((8, 8, 3), 128, dtype=np.uint8)
        gamma_corrected = preprocess_phase0.apply_gamma_correction(frame, gamma=0.5)
        self.assertFalse(np.array_equal(frame, gamma_corrected))

        bright = np.full((8, 8, 3), 255, dtype=np.uint8)
        suppressed = preprocess_phase0.suppress_highlights(bright, threshold=200)
        self.assertLessEqual(int(suppressed.max()), 200)

        clahe_out = preprocess_phase0.apply_clahe(frame)
        self.assertEqual(clahe_out.shape, frame.shape)

    def test_quality_metrics_and_filter_quality(self):
        checker = np.indices((32, 32)).sum(axis=0) % 2
        checker = (checker * 255).astype(np.uint8)
        frame = np.stack([checker, checker, checker], axis=-1)
        metrics = preprocess_phase0.get_frame_quality_metrics(frame)
        self.assertIn("laplacian_var", metrics)
        self.assertGreater(metrics["laplacian_var"], 0)

        passed, reason = preprocess_phase0.filter_frame_quality(
            frame, blur_threshold=1, brightness_low=10, brightness_high=240
        )
        self.assertTrue(passed)
        self.assertEqual(reason, "pass")

        dark_pattern = np.where(checker > 0, 20, 0).astype(np.uint8)
        dark = np.stack([dark_pattern, dark_pattern, dark_pattern], axis=-1)
        passed, reason = preprocess_phase0.filter_frame_quality(dark, blur_threshold=1, brightness_low=30)
        self.assertFalse(passed)
        self.assertEqual(reason, "brightness")

        flat = np.full((32, 32, 3), 128, dtype=np.uint8)
        passed, reason = preprocess_phase0.filter_frame_quality(flat, blur_threshold=1)
        self.assertFalse(passed)
        self.assertEqual(reason, "blur")

    def test_sample_validation_set_outputs_uniform_subset(self):
        with workspace_tempdir("preprocess_val_") as tmp:
            cleaned_dir = tmp / "frames_cleaned"
            val_dir = tmp / "frames_val"
            cleaned_dir.mkdir()
            for idx in range(60):
                image = np.full((12, 12, 3), idx, dtype=np.uint8)
                cv2.imwrite(str(cleaned_dir / f"frame_{idx:06d}.jpg"), image)

            with mock.patch("builtins.print"):
                stats = preprocess_phase0.sample_validation_set(
                    cleaned_frames_dir=cleaned_dir,
                    val_output_dir=val_dir,
                    sample_ratio=0.2,
                )

            self.assertEqual(stats["total_cleaned"], 60)
            self.assertEqual(stats["sampled_frames"], 50)
            self.assertEqual(len(list(val_dir.glob("frame_*.jpg"))), 50)

    def test_preprocess_phase0_returns_none_when_video_cannot_open(self):
        with workspace_tempdir("preprocess_open_fail_") as tmp:
            fake_capture = FakeVideoCapture(opened=False)
            with (
                mock.patch.object(preprocess_phase0.cv2, "VideoCapture", return_value=fake_capture),
                mock.patch("builtins.print"),
            ):
                stats = preprocess_phase0.preprocess_phase0(
                    video_path=tmp / "missing.mp4",
                    output_dir=tmp / "frames_cleaned",
                )

            self.assertIsNone(stats)

    def test_preprocess_phase0_extracts_filters_and_writes_stats(self):
        with workspace_tempdir("preprocess_main_") as tmp:
            frames = [
                np.full((16, 16, 3), 80, dtype=np.uint8),
                np.full((16, 16, 3), 120, dtype=np.uint8),
                np.full((16, 16, 3), 200, dtype=np.uint8),
            ]
            fake_capture = FakeVideoCapture(frames=frames, fps=1.0)
            output_dir = tmp / "frames_cleaned"

            with (
                mock.patch.object(preprocess_phase0.cv2, "VideoCapture", return_value=fake_capture),
                mock.patch.object(preprocess_phase0, "tqdm", DummyProgress),
                mock.patch.object(
                    preprocess_phase0,
                    "filter_frame_quality",
                    side_effect=[(True, "pass"), (False, "blur"), (False, "brightness")],
                ),
                mock.patch.object(preprocess_phase0, "apply_gamma_correction", side_effect=lambda frame, gamma: frame),
                mock.patch.object(preprocess_phase0, "apply_clahe", side_effect=lambda frame: frame),
                mock.patch.object(preprocess_phase0, "suppress_highlights", side_effect=lambda frame, threshold: frame),
                mock.patch("builtins.print"),
            ):
                stats = preprocess_phase0.preprocess_phase0(
                    video_path=tmp / "input.mp4",
                    output_dir=output_dir,
                    fps=1,
                    gamma=0.7,
                    blur_threshold=5,
                )

            self.assertEqual(stats["accepted_frames"], 1)
            self.assertTrue(fake_capture.released)
            self.assertTrue((output_dir / "frame_000000.jpg").exists())
            self.assertTrue((output_dir / "extraction_stats.json").exists())
            self.assertTrue((output_dir / "filtering_stats.json").exists())
            self.assertEqual(stats["extraction_stats"]["extracted_frames"], 3)
            self.assertEqual(stats["extraction_stats"]["status"], "FAIL")
            self.assertEqual(stats["filtering_stats"]["accepted_frames"], 1)
            self.assertEqual(stats["filtering_stats"]["rejected_reasons"], {"blur": 1, "brightness": 1})
            self.assertEqual(stats["filtering_stats"]["status"], "FAIL")

    def test_downscale_main_resizes_to_requested_max_side(self):
        with workspace_tempdir("downscale_") as tmp:
            src_dir = tmp / "src"
            dst_dir = tmp / "dst"
            src_dir.mkdir()
            image = np.full((1000, 2000, 3), 127, dtype=np.uint8)
            cv2.imwrite(str(src_dir / "frame_000000.jpg"), image)

            with mock.patch.object(downscale_frames, "tqdm", lambda items, **_: items), mock.patch("builtins.print"):
                downscale_frames.main(src=str(src_dir), dst=str(dst_dir), max_side=1000)

            out = cv2.imread(str(dst_dir / "frame_000000.jpg"))
            self.assertEqual(max(out.shape[:2]), 1000)


if __name__ == "__main__":
    unittest.main()

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

from src import downscale_frames, preprocess_phase0


class PreprocessAndDownscaleTests(unittest.TestCase):
    def test_gamma_and_highlight_helpers_change_pixels(self):
        frame = np.full((8, 8, 3), 128, dtype=np.uint8)
        gamma_corrected = preprocess_phase0.apply_gamma_correction(frame, gamma=0.5)
        self.assertFalse(np.array_equal(frame, gamma_corrected))

        bright = np.full((8, 8, 3), 255, dtype=np.uint8)
        suppressed = preprocess_phase0.suppress_highlights(bright, threshold=200)
        self.assertLessEqual(int(suppressed.max()), 200)

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

    def test_sample_validation_set_outputs_uniform_subset(self):
        with tempfile.TemporaryDirectory() as tmp:
            cleaned_dir = Path(tmp) / "frames_cleaned"
            val_dir = Path(tmp) / "frames_val"
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

    def test_downscale_main_resizes_to_requested_max_side(self):
        with tempfile.TemporaryDirectory() as tmp:
            src_dir = Path(tmp) / "src"
            dst_dir = Path(tmp) / "dst"
            src_dir.mkdir()
            image = np.full((1000, 2000, 3), 127, dtype=np.uint8)
            cv2.imwrite(str(src_dir / "frame_000000.jpg"), image)

            with mock.patch.object(downscale_frames, "tqdm", lambda items, **_: items), mock.patch("builtins.print"):
                downscale_frames.main(src=str(src_dir), dst=str(dst_dir), max_side=1000)

            out = cv2.imread(str(dst_dir / "frame_000000.jpg"))
            self.assertEqual(max(out.shape[:2]), 1000)


if __name__ == "__main__":
    unittest.main()

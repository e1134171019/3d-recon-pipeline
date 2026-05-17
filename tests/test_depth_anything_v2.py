import unittest

from pathlib import Path

import cv2
import numpy as np

from src.depth_anything_v2 import (
    DEPTH_ANYTHING_V2_UNAVAILABLE,
    DepthAnythingV2Estimator,
    compute_depth_consistency_loss,
)
from _workspace_temp import workspace_tempdir


class DepthAnythingV2GuardTests(unittest.TestCase):
    def test_estimate_depth_raises_file_not_found_for_missing_image(self):
        estimator = DepthAnythingV2Estimator()
        with self.assertRaises(FileNotFoundError):
            estimator.estimate_depth(Path("missing_image.png"))

    def test_estimate_batch_depth_raises_file_not_found_for_empty_dir(self):
        estimator = DepthAnythingV2Estimator()
        with workspace_tempdir("depth_anything_empty_") as tmp:
            with self.assertRaises(FileNotFoundError):
                estimator.estimate_batch_depth(tmp)

    def test_estimate_depth_raises_not_implemented_for_readable_image(self):
        estimator = DepthAnythingV2Estimator()
        with workspace_tempdir("depth_anything_img_") as tmp:
            image_path = tmp / "frame_0001.png"
            image = np.full((8, 8, 3), 127, dtype=np.uint8)
            ok = cv2.imwrite(str(image_path), image)
            self.assertTrue(ok)

            with self.assertRaises(NotImplementedError) as raised:
                estimator.estimate_depth(image_path)

        self.assertIn(DEPTH_ANYTHING_V2_UNAVAILABLE, str(raised.exception))

    def test_compute_depth_consistency_loss_returns_weighted_mean_abs_error(self):
        predicted = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32)
        estimated = np.array([[0.0, 2.0], [1.0, 3.0]], dtype=np.float32)
        loss = compute_depth_consistency_loss(
            predicted_depth=predicted,
            estimated_depth=estimated,
            lambda_weight=0.5,
        )
        self.assertGreater(loss, 0.0)
        self.assertLess(loss, 0.5)


if __name__ == "__main__":
    unittest.main()

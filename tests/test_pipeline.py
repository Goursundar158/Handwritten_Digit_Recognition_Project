"""Smoke tests for preprocessing and feature extraction."""

from __future__ import annotations

import unittest

import numpy as np
from sklearn.datasets import load_digits

from src.config import IMAGE_SIZE
from src.features import hog_feature, make_features
from src.preprocessing import preprocess_digit, preprocess_digit_stages


class PipelineTests(unittest.TestCase):
    """Verify the core image-processing steps."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.digits = load_digits()

    def test_preprocess_shape_and_range(self) -> None:
        output = preprocess_digit(self.digits.images[0], size=IMAGE_SIZE)
        self.assertEqual(output.shape, (IMAGE_SIZE, IMAGE_SIZE))
        self.assertEqual(output.dtype, np.uint8)
        self.assertGreaterEqual(int(output.min()), 0)
        self.assertLessEqual(int(output.max()), 255)
        self.assertTrue(np.any(output > 0))

    def test_default_stage_names(self) -> None:
        stages = preprocess_digit_stages(self.digits.images[4], size=IMAGE_SIZE)
        self.assertEqual(set(stages), {"Input", "Polarity fixed", "Resized"})

    def test_full_pipeline_stages(self) -> None:
        stages = preprocess_digit_stages(
            self.digits.images[4],
            size=IMAGE_SIZE,
            use_denoising=True,
            use_binarization=True,
            use_crop=True,
            use_deskew=True,
        )
        for name in (
            "Input",
            "Denoised",
            "Binarized",
            "Polarity fixed",
            "Cropped",
            "Padded",
            "Deskewed",
            "Resized",
        ):
            self.assertIn(name, stages)

    def test_hog_descriptor_length(self) -> None:
        image = preprocess_digit(self.digits.images[7], size=IMAGE_SIZE)
        descriptor = hog_feature(image, cell_size=8, bins=9)
        # 4x4 cells, 3x3 overlapping blocks, 4 cells per block, 9 bins.
        self.assertEqual(descriptor.shape, (324,))

    def test_feature_variants(self) -> None:
        batch = self.digits.images[:5]
        expected = {"raw8": 64, "gray32": 1024, "hog32": 324, "hybrid": 580}
        for variant, width in expected.items():
            with self.subTest(variant=variant):
                features = make_features(batch, variant=variant)
                self.assertEqual(features.shape, (5, width))


if __name__ == "__main__":
    unittest.main()

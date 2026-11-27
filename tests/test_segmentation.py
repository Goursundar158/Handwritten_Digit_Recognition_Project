"""Tests for connected-component digit segmentation."""

from __future__ import annotations

import unittest

import cv2
import numpy as np

from src.generate_samples import generate_form_sample
from src.segmentation import DEFAULT_CANVAS_SIZE, segment_digits


class SegmentationTests(unittest.TestCase):
    """Verify that a form-style image is split into separate digits."""

    def test_form_segmentation(self) -> None:
        path, sequence = generate_form_sample(sequence="3014")
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        self.assertIsNotNone(image)

        segments = segment_digits(image)
        self.assertEqual(len(segments), len(sequence))
        for segment in segments:
            x, y, width, height = segment["bbox"]
            self.assertGreater(width, 3)
            self.assertGreater(height, 5)
            self.assertEqual(
                segment["image"].shape, (DEFAULT_CANVAS_SIZE, DEFAULT_CANVAS_SIZE)
            )

    def test_empty_image_has_no_segments(self) -> None:
        blank = np.zeros((64, 64), dtype=np.uint8)
        self.assertEqual(segment_digits(blank), [])


if __name__ == "__main__":
    unittest.main()

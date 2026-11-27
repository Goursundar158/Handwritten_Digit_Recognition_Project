"""Feature extraction for normalised digit images.

Two representations are implemented:

1. Downsampled grayscale pixels (a simple baseline).
2. Histogram of Oriented Gradients (HOG), implemented with NumPy so the
   project has no hidden feature-extractor dependency beyond OpenCV/NumPy.
"""

from __future__ import annotations

import cv2
import numpy as np

from .preprocessing import preprocess_batch, resize_image


def _cell_histograms(
    magnitude: np.ndarray,
    orientation: np.ndarray,
    cell_size: int,
    bins: int,
) -> np.ndarray:
    """Build an orientation histogram for every spatial cell."""
    height, width = magnitude.shape
    cells_y = height // cell_size
    cells_x = width // cell_size
    histograms = np.zeros((cells_y, cells_x, bins), dtype=np.float32)
    bin_width = 180.0 / bins

    for cy in range(cells_y):
        for cx in range(cells_x):
            ys = slice(cy * cell_size, (cy + 1) * cell_size)
            xs = slice(cx * cell_size, (cx + 1) * cell_size)
            cell_mag = magnitude[ys, xs].ravel()
            cell_ang = orientation[ys, xs].ravel()

            positions = cell_ang / bin_width
            lower = np.floor(positions).astype(np.int64)
            fraction = positions - lower
            lower %= bins
            upper = (lower + 1) % bins

            np.add.at(histograms[cy, cx], lower, cell_mag * (1.0 - fraction))
            np.add.at(histograms[cy, cx], upper, cell_mag * fraction)

    return histograms


def hog_feature(image: np.ndarray, cell_size: int = 8, bins: int = 9) -> np.ndarray:
    """Compute a normalized HOG descriptor for one grayscale image."""
    image_f = image.astype(np.float32) / 255.0
    if image_f.shape[0] % cell_size or image_f.shape[1] % cell_size:
        raise ValueError("image dimensions must be divisible by cell_size")

    gx = cv2.Sobel(image_f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(image_f, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = np.sqrt(gx * gx + gy * gy)
    orientation = (np.degrees(np.arctan2(gy, gx)) % 180.0).astype(np.float32)

    histograms = _cell_histograms(magnitude, orientation, cell_size, bins)
    block_features: list[np.ndarray] = []
    for by in range(histograms.shape[0] - 1):
        for bx in range(histograms.shape[1] - 1):
            block = histograms[by : by + 2, bx : bx + 2].ravel()
            # L2-Hys style clipping improves robustness to stroke thickness.
            block = np.minimum(block, 0.2)
            norm = np.linalg.norm(block) + 1e-6
            block_features.append(block / norm)

    return np.concatenate(block_features).astype(np.float32)


def raw_pixel_features(images, image_size: int = 32) -> np.ndarray:
    """Flatten images into raw-pixel feature vectors after size normalization."""
    return np.asarray(
        [
            resize_image(image, size=image_size, force_binary=False).ravel()
            for image in images
        ],
        dtype=np.float32,
    ) / 255.0


def make_features(
    images,
    variant: str = "hybrid",
    image_size: int = 32,
    pixel_size: int = 16,
    cell_size: int = 8,
    bins: int = 9,
) -> np.ndarray:
    """Build feature vectors according to a named variant.

    Variants:
        raw8: original 8x8 grayscale pixels (baseline, no handcrafted features)
        gray32: grayscale image resized to 32x32
        hog32: HOG descriptor from a 32x32 grayscale image
        hybrid: 16x16 grayscale pixels concatenated with 32x32 HOG descriptor
    """
    if variant == "raw8":
        return raw_pixel_features(images, image_size=8)

    processed = preprocess_batch(images, size=image_size)
    if variant == "gray32":
        return raw_pixel_features(processed, image_size=image_size)

    hog_features = np.asarray(
        [hog_feature(img, cell_size=cell_size, bins=bins) for img in processed],
        dtype=np.float32,
    )
    if variant == "hog32":
        return hog_features
    if variant == "hybrid":
        pixel_features = raw_pixel_features(processed, image_size=pixel_size)
        return np.hstack([pixel_features, hog_features]).astype(np.float32)

    raise ValueError(f"Unknown feature variant: {variant}")

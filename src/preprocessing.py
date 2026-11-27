"""Image-preprocessing functions for isolated handwritten digits.

The default path used for model training is deliberately light: grayscale
conversion and normalized resizing. Stronger operations (median denoising,
Otsu binarization, cropping and moment-based deskewing) are available for
degraded inputs and are demonstrated in results/preprocessing_pipeline.png.
"""

from __future__ import annotations

import cv2
import numpy as np


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert BGR/RGB/grayscale input to an 8-bit grayscale image."""
    arr = np.asarray(image)
    if arr.ndim == 3:
        if arr.shape[2] == 4:
            arr = cv2.cvtColor(arr, cv2.COLOR_BGRA2GRAY)
        else:
            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)

    if arr.dtype != np.uint8:
        arr = arr.astype(np.float32)
        if arr.size and arr.max() <= 1.0:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def denoise(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Apply median filtering to suppress impulse noise."""
    if kernel_size < 3 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be an odd integer >= 3")
    return cv2.medianBlur(image, kernel_size)


def binarize(image: np.ndarray) -> np.ndarray:
    """Apply Otsu's global threshold method."""
    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def invert_if_needed(binary: np.ndarray) -> np.ndarray:
    """Return an image with white foreground on a black background."""
    if binary.size == 0:
        return binary
    # A digit occupies a minority of the canvas. If most pixels are bright,
    # the input has a white background and must be inverted.
    if float(np.mean(binary > 0)) > 0.5:
        return cv2.bitwise_not(binary)
    return binary


def crop_to_content(binary: np.ndarray, padding: int = 1) -> np.ndarray:
    """Crop to the tight bounding box around non-zero pixels."""
    points = cv2.findNonZero(binary)
    if points is None:
        return np.zeros_like(binary)
    x, y, w, h = cv2.boundingRect(points)
    x0 = max(0, x - padding)
    y0 = max(0, y - padding)
    x1 = min(binary.shape[1], x + w + padding)
    y1 = min(binary.shape[0], y + h + padding)
    return binary[y0:y1, x0:x1]


def pad_to_square(image: np.ndarray, margin: int = 3) -> np.ndarray:
    """Center an image in a square canvas while preserving its aspect ratio."""
    h, w = image.shape[:2]
    side = int(max(h, w) + 2 * margin)
    canvas = np.zeros((side, side), dtype=np.uint8)
    y = (side - h) // 2
    x = (side - w) // 2
    canvas[y : y + h, x : x + w] = image
    return canvas


def deskew(image: np.ndarray) -> np.ndarray:
    """Correct small rotations using second-order image moments."""
    moments = cv2.moments(image)
    if abs(moments["mu02"]) < 1e-2:
        return image
    skew = moments["mu11"] / moments["mu02"]
    angle = float(np.degrees(np.arctan(skew)))
    # Limit correction so unusual shapes cannot be rotated dramatically.
    angle = float(np.clip(angle, -25.0, 25.0))
    center = (image.shape[1] / 2.0, image.shape[0] / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(
        image,
        matrix,
        (image.shape[1], image.shape[0]),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def resize_image(
    image: np.ndarray,
    size: int = 32,
    force_binary: bool = False,
) -> np.ndarray:
    """Resize an image to a fixed square size."""
    interpolation = cv2.INTER_AREA if max(image.shape[:2]) > size else cv2.INTER_CUBIC
    resized = cv2.resize(image, (size, size), interpolation=interpolation)
    if force_binary:
        return (resized > 0).astype(np.uint8) * 255
    return np.clip(resized, 0, 255).astype(np.uint8)


def preprocess_digit(
    image: np.ndarray,
    size: int = 32,
    use_denoising: bool = False,
    use_binarization: bool = False,
    use_crop: bool = False,
    use_deskew: bool = False,
) -> np.ndarray:
    """Convert one digit image into a normalized 8-bit image.

    Light preprocessing is the default because the scikit-learn digits are
    already clean and centered. The optional operations are useful for
    photographs and scanned forms with noise, background or rotation.
    """
    gray = to_grayscale(image)
    if use_denoising:
        gray = denoise(gray)

    if not (use_binarization or use_crop or use_deskew):
        return resize_image(gray, size=size)

    if use_binarization:
        working = binarize(gray)
    else:
        working = (gray > 0).astype(np.uint8) * 255
    working = invert_if_needed(working)

    if use_crop:
        working = crop_to_content(working)
        if not np.any(working):
            return np.zeros((size, size), dtype=np.uint8)

    if use_deskew:
        working = pad_to_square(working, margin=3)
        working = deskew(working)

    return resize_image(working, size=size, force_binary=True)


def preprocess_digit_stages(
    image: np.ndarray,
    size: int = 32,
    use_denoising: bool = False,
    use_binarization: bool = False,
    use_crop: bool = False,
    use_deskew: bool = False,
) -> dict[str, np.ndarray]:
    """Return intermediate images so the pipeline can be visualized."""
    gray = to_grayscale(image)
    stages: dict[str, np.ndarray] = {"Input": gray}

    if use_denoising:
        gray = denoise(gray)
        stages["Denoised"] = gray

    if use_binarization:
        gray = binarize(gray)
        stages["Binarized"] = gray

    working = invert_if_needed(gray)
    stages["Polarity fixed"] = working

    if use_crop:
        working = crop_to_content(working)
        stages["Cropped"] = working
        if not np.any(working):
            stages["Resized"] = np.zeros((size, size), dtype=np.uint8)
            return stages

    if use_deskew:
        working = pad_to_square(working, margin=3)
        stages["Padded"] = working
        working = deskew(working)
        stages["Deskewed"] = working

    stages["Resized"] = resize_image(
        working,
        size=size,
        force_binary=use_binarization or use_crop or use_deskew,
    )
    return stages


def preprocess_batch(images, size: int = 32, **kwargs) -> np.ndarray:
    """Preprocess a list/array of digit images into a numeric array."""
    return np.asarray(
        [preprocess_digit(img, size=size, **kwargs) for img in images],
        dtype=np.uint8,
    )

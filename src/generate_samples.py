"""Create example digit images from the built-in scikit-learn digits data."""

from __future__ import annotations

import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits

from .config import PIPELINE_FIGURE_PATH, RESULTS_DIR, SAMPLES_DIR
from .preprocessing import preprocess_digit_stages


def _save_image(path, image: np.ndarray) -> None:
    cv2.imwrite(str(path), image)


def generate_sample_images(count_per_class: int = 2) -> list[dict[str, object]]:
    """Save enlarged digit PNGs and return their metadata."""
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    digits = load_digits()
    metadata: list[dict[str, object]] = []

    for label in range(10):
        indices = np.flatnonzero(digits.target == label)[:count_per_class]
        for number, index in enumerate(indices, start=1):
            enlarged = cv2.resize(
                digits.images[index], (128, 128), interpolation=cv2.INTER_CUBIC
            )
            image_u8 = np.clip(enlarged, 0, 255).astype(np.uint8)
            path = SAMPLES_DIR / f"digit_{label}_{number}.png"
            _save_image(path, image_u8)
            metadata.append(
                {"path": str(path), "label": int(label), "dataset_index": int(index)}
            )

    # A deliberately degraded image demonstrates denoising and deskewing.
    base_index = int(np.flatnonzero(digits.target == 8)[0])
    base = np.clip(digits.images[base_index], 0, 255).astype(np.uint8)
    large = cv2.resize(base, (128, 128), interpolation=cv2.INTER_CUBIC)
    center = (64, 64)
    rotation = cv2.getRotationMatrix2D(center, 3.0, 1.0)
    rotated = cv2.warpAffine(large, rotation, (128, 128), borderValue=0)
    noise = np.random.default_rng(0).normal(0, 5, rotated.shape).astype(np.int16)
    noisy = np.clip(rotated.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    degraded_path = SAMPLES_DIR / "digit_8_degraded.png"
    _save_image(degraded_path, noisy)
    metadata.append(
        {
            "path": str(degraded_path),
            "label": 8,
            "dataset_index": base_index,
            "note": "rotated and noisy demonstration image",
        }
    )

    return metadata


def generate_form_sample(sequence: str = "3014", spacing: int = 24):
    """Build a form-style image containing several separated digits."""
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    if not (SAMPLES_DIR / f"digit_{sequence[0]}_1.png").exists():
        generate_sample_images()

    digit_images = []
    for digit in sequence:
        path = SAMPLES_DIR / f"digit_{digit}_1.png"
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Missing sample image: {path}")
        digit_images.append(image)

    height = max(image.shape[0] for image in digit_images)
    width = sum(image.shape[1] for image in digit_images)
    width += spacing * (len(digit_images) - 1) + 20
    canvas = np.zeros((height, width), dtype=np.uint8)

    x = 10
    for image in digit_images:
        y = (height - image.shape[0]) // 2
        canvas[y : y + image.shape[0], x : x + image.shape[1]] = image
        x += image.shape[1] + spacing

    noise = np.random.default_rng(3).normal(0, 4, canvas.shape).astype(np.int16)
    form = np.clip(canvas.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    output_path = SAMPLES_DIR / f"form_{sequence}.png"
    _save_image(output_path, form)
    return output_path, sequence


def generate_pipeline_figure() -> None:
    """Save a figure showing every preprocessing stage for the degraded image."""
    source = SAMPLES_DIR / "digit_8_degraded.png"
    if not source.exists():
        generate_sample_images()

    image = cv2.imread(str(source), cv2.IMREAD_GRAYSCALE)
    stages = preprocess_digit_stages(
        image,
        size=32,
        use_denoising=True,
        use_binarization=True,
        use_crop=True,
        use_deskew=True,
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(2, 4, figsize=(10, 5.5))
    for axis, (name, stage) in zip(axes.ravel(), stages.items()):
        axis.imshow(stage, cmap="gray", vmin=0, vmax=255)
        axis.set_title(name)
        axis.axis("off")
    for axis in axes.ravel()[len(stages) :]:
        axis.axis("off")
    figure.suptitle("Digit preprocessing pipeline")
    figure.tight_layout()
    figure.savefig(PIPELINE_FIGURE_PATH, dpi=180)
    plt.close(figure)


if __name__ == "__main__":
    for item in generate_sample_images():
        print(item["path"])
    generate_pipeline_figure()
    print(f"Saved pipeline figure to {PIPELINE_FIGURE_PATH}")
    form_path, sequence = generate_form_sample()
    print(f"Saved form sample '{sequence}' to {form_path}")

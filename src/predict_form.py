r"""Segment and classify a form-style image containing several digits.

Examples (from the project root):
    python -m src.predict_form --image data\samples\form_3014.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from . import config
from .features import make_features
from .predict import load_artifact
from .segmentation import draw_segmentation_boxes, segment_digits


def predict_form(
    image_path: str | Path,
    save_figure: bool = False,
) -> dict[str, object]:
    """Detect digit components in an image and classify each one."""
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    segments = segment_digits(image)
    if not segments:
        result = {
            "image": str(image_path),
            "count": 0,
            "digits": "",
            "details": [],
        }
        if save_figure:
            result["figure"] = str(config.SEGMENTATION_FIGURE_PATH)
        return result

    artifact = load_artifact()
    features = make_features(
        [segment["image"] for segment in segments],
        variant=artifact["feature_variant"],
        image_size=artifact["image_size"],
        pixel_size=artifact["pixel_size"],
        cell_size=artifact["cell_size"],
        bins=artifact["bins"],
    )
    scaled = artifact["scaler"].transform(features)
    predictions = [int(value) for value in artifact["classifier"].predict(scaled)]

    confidences: list[float | None] = [None] * len(segments)
    if hasattr(artifact["classifier"], "predict_proba"):
        probabilities = artifact["classifier"].predict_proba(scaled)
        confidences = [float(np.max(row)) for row in probabilities]

    details = []
    for segment, prediction, confidence in zip(segments, predictions, confidences):
        details.append(
            {
                "bbox": list(segment["bbox"]),
                "predicted_digit": prediction,
                "confidence": confidence,
            }
        )

    result = {
        "image": str(image_path),
        "count": len(segments),
        "digits": "".join(str(digit) for digit in predictions),
        "details": details,
    }

    if save_figure:
        config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        boxed = draw_segmentation_boxes(image, segments)
        cv2.imwrite(str(config.SEGMENTATION_FIGURE_PATH), boxed)
        result["figure"] = str(config.SEGMENTATION_FIGURE_PATH)

    return result


def main() -> None:
    """Command-line interface for form-style digit recognition."""
    parser = argparse.ArgumentParser(
        description="Segment and recognize a string of handwritten digits."
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=config.SAMPLES_DIR / "form_3014.png",
        help="Path to a form image containing separated handwritten digits.",
    )
    parser.add_argument(
        "--save-figure",
        action="store_true",
        help="Save results/segmentation_boxes.png with detected boxes.",
    )
    args = parser.parse_args()

    result = predict_form(args.image, save_figure=args.save_figure)
    print(f"Image              : {result['image']}")
    print(f"Digits segmented   : {result['count']}")
    print(f"Recognized string  : {result['digits'] or '(none)'}")
    for index, detail in enumerate(result["details"], start=1):
        confidence = detail["confidence"]
        confidence_text = f"{confidence:.3f}" if confidence is not None else "n/a"
        print(
            f"  {index}: digit={detail['predicted_digit']} "
            f"confidence={confidence_text} bbox={detail['bbox']}"
        )
    if "figure" in result:
        print(f"Segmentation figure: {result['figure']}")


if __name__ == "__main__":
    main()

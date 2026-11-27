"""Predict a digit from an image using the saved trained model.

Examples (from the project root):
    python -m src.predict --image data/samples/digit_8_degraded.png
    python -m src.predict --image data/samples/digit_3_1.png --save-preprocessed
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import joblib
import numpy as np

from . import config
from .features import make_features
from .preprocessing import preprocess_digit


def load_artifact() -> dict:
    """Load the trained model bundle produced by src.train."""
    if not config.MODEL_PATH.exists():
        raise FileNotFoundError(
            f"No trained model found at {config.MODEL_PATH}. "
            "Run 'python -m src.train' first."
        )
    return joblib.load(config.MODEL_PATH)


def predict_image(image_path: str | Path, save_preprocessed: bool = False) -> dict:
    """Preprocess, featurize, and classify one digit image."""
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    artifact = load_artifact()
    features = make_features(
        [image],
        variant=artifact["feature_variant"],
        image_size=artifact["image_size"],
        pixel_size=artifact["pixel_size"],
        cell_size=artifact["cell_size"],
        bins=artifact["bins"],
    )
    scaled = artifact["scaler"].transform(features)
    prediction = int(artifact["classifier"].predict(scaled)[0])

    confidence = None
    if hasattr(artifact["classifier"], "predict_proba"):
        probabilities = artifact["classifier"].predict_proba(scaled)[0]
        confidence = float(np.max(probabilities))

    result = {
        "image": str(image_path),
        "predicted_digit": prediction,
        "confidence": confidence,
        "model": artifact["model_name"],
        "features": artifact["feature_variant"],
    }

    if save_preprocessed:
        output = config.RESULTS_DIR / "preprocessed_input.png"
        output.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(
            str(output),
            preprocess_digit(image, size=artifact["image_size"]),
        )
        result["preprocessed_image"] = str(output)

    return result


def main() -> None:
    """Command-line interface for single-image prediction."""
    parser = argparse.ArgumentParser(description="Recognize one handwritten digit.")
    parser.add_argument(
        "--image",
        type=Path,
        default=config.SAMPLES_DIR / "digit_8_degraded.png",
        help="Path to a digit image (grayscale or color).",
    )
    parser.add_argument(
        "--save-preprocessed",
        action="store_true",
        help="Also save results/preprocessed_input.png.",
    )
    args = parser.parse_args()

    result = predict_image(args.image, save_preprocessed=args.save_preprocessed)
    print(f"Image              : {result['image']}")
    print(f"Predicted digit    : {result['predicted_digit']}")
    if result["confidence"] is not None:
        print(f"Confidence         : {result['confidence']:.3f}")
    print(f"Model / features   : {result['model']} / {result['features']}")
    if "preprocessed_image" in result:
        print(f"Preprocessed image : {result['preprocessed_image']}")


if __name__ == "__main__":
    main()

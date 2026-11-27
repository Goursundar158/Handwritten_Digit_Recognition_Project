"""Train and evaluate handwritten digit recognition models.

Run from the project root:
    python -m src.train
"""

from __future__ import annotations

import json
import time

import matplotlib

matplotlib.use("Agg")

import cv2
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from . import config
from .features import make_features
from .generate_samples import (
    generate_form_sample,
    generate_pipeline_figure,
    generate_sample_images,
)
from .predict_form import predict_form

FEATURE_VARIANTS = ("raw8", "gray32", "hog32", "hybrid")


def build_models() -> dict[str, object]:
    """Return the classifiers compared in the experiment."""
    return {
        "k-NN (k=5)": KNeighborsClassifier(n_neighbors=5),
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=config.RANDOM_STATE),
        "SVM (RBF)": SVC(
            C=10.0,
            gamma="scale",
            kernel="rbf",
            probability=True,
            random_state=config.RANDOM_STATE,
        ),
    }


def plot_confusion_matrix(matrix: np.ndarray, output_path) -> None:
    """Save a labeled confusion-matrix heatmap."""
    figure, axis = plt.subplots(figsize=(7, 6))
    image = axis.imshow(matrix, interpolation="nearest", cmap=plt.cm.Blues)
    figure.colorbar(image, ax=axis)
    axis.set(
        xlabel="Predicted digit",
        ylabel="True digit",
        title="Confusion matrix - best model",
    )
    axis.set_xticks(np.arange(10))
    axis.set_yticks(np.arange(10))
    threshold = matrix.max() / 2.0 if matrix.size else 0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            axis.text(
                j,
                i,
                str(matrix[i, j]),
                ha="center",
                va="center",
                color="white" if matrix[i, j] > threshold else "black",
                fontsize=8,
            )
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def main() -> None:
    """Run all experiments, persist metrics, and save the best model."""
    for directory in (config.RESULTS_DIR, config.MODELS_DIR):
        directory.mkdir(parents=True, exist_ok=True)

    digits = load_digits()
    images = digits.images
    labels = digits.target

    indices = np.arange(len(labels))
    train_idx, test_idx = train_test_split(
        indices,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=labels,
    )
    y_train, y_test = labels[train_idx], labels[test_idx]

    experiment_results: list[dict[str, object]] = []
    feature_cache: dict[str, np.ndarray] = {}

    for variant in FEATURE_VARIANTS:
        features = make_features(
            images,
            variant=variant,
            image_size=config.IMAGE_SIZE,
            pixel_size=config.PIXEL_SIZE,
            cell_size=config.HOG_CELL_SIZE,
            bins=config.HOG_BINS,
        )
        feature_cache[variant] = features
        x_train, x_test = features[train_idx], features[test_idx]

        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(x_train)
        x_test_scaled = scaler.transform(x_test)

        for model_name, estimator in build_models().items():
            started = time.perf_counter()
            fitted = clone(estimator).fit(x_train_scaled, y_train)
            predictions = fitted.predict(x_test_scaled)
            elapsed = time.perf_counter() - started

            experiment_results.append(
                {
                    "feature_variant": variant,
                    "model": model_name,
                    "accuracy": float(accuracy_score(y_test, predictions)),
                    "macro_f1": float(f1_score(y_test, predictions, average="macro")),
                    "train_seconds": round(elapsed, 4),
                    "feature_count": int(features.shape[1]),
                }
            )
            print(
                f"{variant:9s} | {model_name:20s} | "
                f"acc={experiment_results[-1]['accuracy']:.4f} | "
                f"macro-F1={experiment_results[-1]['macro_f1']:.4f}"
            )

    best = max(experiment_results, key=lambda row: (row["accuracy"], row["macro_f1"]))
    best_variant = str(best["feature_variant"])
    best_model_name = str(best["model"])

    best_features = feature_cache[best_variant]
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(best_features[train_idx])
    x_test_scaled = scaler.transform(best_features[test_idx])
    classifier = clone(build_models()[best_model_name]).fit(x_train_scaled, y_train)
    y_pred = classifier.predict(x_test_scaled)

    report = classification_report(y_test, y_pred, output_dict=True, digits=4)
    matrix = confusion_matrix(y_test, y_pred, labels=np.arange(10))

    artifact = {
        "scaler": scaler,
        "classifier": classifier,
        "feature_variant": best_variant,
        "model_name": best_model_name,
        "image_size": config.IMAGE_SIZE,
        "pixel_size": config.PIXEL_SIZE,
        "cell_size": config.HOG_CELL_SIZE,
        "bins": config.HOG_BINS,
        "classes": np.arange(10),
    }
    joblib.dump(artifact, config.MODEL_PATH)

    sample_metadata = generate_sample_images()
    generate_pipeline_figure()
    form_path, expected_form = generate_form_sample()
    form_result = predict_form(form_path, save_figure=True)

    metrics = {
        "dataset": {
            "name": "scikit-learn digits (8x8 handwritten digit images)",
            "samples": int(len(labels)),
            "classes": 10,
            "train_size": int(len(train_idx)),
            "test_size": int(len(test_idx)),
            "test_fraction": config.TEST_SIZE,
            "random_state": config.RANDOM_STATE,
            "stratified_split": True,
        },
        "experiments": sorted(
            experiment_results,
            key=lambda row: (row["accuracy"], row["macro_f1"]),
            reverse=True,
        ),
        "best": best,
        "classification_report": report,
        "confusion_matrix": matrix.tolist(),
        "form_segmentation": {
            "image": str(form_path),
            "expected": expected_form,
            "predicted": form_result["digits"],
            "segments": int(form_result["count"]),
            "correct": form_result["digits"] == expected_form,
        },
    }
    with open(config.METRICS_PATH, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    plot_confusion_matrix(matrix, config.CONFUSION_MATRIX_PATH)

    sample_rows = []
    for row in sample_metadata:
        image_bgr = cv2.imread(str(row["path"]), cv2.IMREAD_GRAYSCALE)
        if image_bgr is None:
            continue
        features = make_features(
            [image_bgr],
            variant=best_variant,
            image_size=config.IMAGE_SIZE,
            pixel_size=config.PIXEL_SIZE,
            cell_size=config.HOG_CELL_SIZE,
            bins=config.HOG_BINS,
        )
        prediction = int(classifier.predict(scaler.transform(features))[0])
        sample_rows.append(
            {
                "image": row["path"],
                "expected_digit": row["label"],
                "predicted_digit": prediction,
            }
        )
    pd.DataFrame(sample_rows).to_csv(config.PREDICTIONS_PATH, index=False)

    print("\nBest configuration:")
    print(json.dumps(best, indent=2))
    print(f"\nSaved model: {config.MODEL_PATH}")
    print(f"Saved metrics: {config.METRICS_PATH}")
    print(f"Saved confusion matrix: {config.CONFUSION_MATRIX_PATH}")
    print(f"Saved segmentation figure: {config.SEGMENTATION_FIGURE_PATH}")
    print(f"Saved sample predictions: {config.PREDICTIONS_PATH}")
    print(
        f"Form test: expected {expected_form}, "
        f"segmented {form_result['count']} digits, "
        f"predicted {form_result['digits']}"
    )


if __name__ == "__main__":
    main()

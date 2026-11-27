"""Connected-component segmentation for form-style digit strings.

The function :func:`segment_digits` accepts a scanned image containing several
isolated handwritten digits, thresholds it, finds connected foreground
components, removes small noise components and returns one cropped image per
digit in left-to-right reading order.
"""

from __future__ import annotations

import cv2
import numpy as np

from .config import IMAGE_SIZE
from .preprocessing import binarize, invert_if_needed, to_grayscale

DEFAULT_CANVAS_SIZE = IMAGE_SIZE * 4


def _center_on_canvas(component: np.ndarray, canvas_size: int) -> np.ndarray:
    """Center a tight digit crop in a square canvas like the training samples."""
    height, width = component.shape[:2]
    if height > canvas_size or width > canvas_size:
        scale = min(canvas_size / width, canvas_size / height)
        new_width = max(1, int(width * scale))
        new_height = max(1, int(height * scale))
        component = cv2.resize(
            component, (new_width, new_height), interpolation=cv2.INTER_AREA
        )
        height, width = component.shape[:2]

    canvas = np.zeros((canvas_size, canvas_size), dtype=np.uint8)
    y = (canvas_size - height) // 2
    x = (canvas_size - width) // 2
    canvas[y : y + height, x : x + width] = component
    return canvas


def _split_on_zero_columns(component: np.ndarray, min_width: int = 4) -> list[np.ndarray]:
    """Split a wide component at completely empty vertical projection gaps."""
    column_sums = (component > 0).sum(axis=0)
    active = column_sums > 0
    groups: list[tuple[int, int]] = []
    start: int | None = None

    for index, value in enumerate(active):
        if value and start is None:
            start = index
        elif not value and start is not None:
            groups.append((start, index))
            start = None
    if start is not None:
        groups.append((start, len(active)))

    parts = [component[:, left:right] for left, right in groups]
    parts = [part for part in parts if part.shape[1] >= min_width]
    return parts if len(parts) > 1 else [component]


def segment_digits(
    image: np.ndarray,
    min_area: int = 40,
    max_digits: int = 20,
    close_kernel_size: int = 3,
    canvas_size: int = DEFAULT_CANVAS_SIZE,
) -> list[dict[str, object]]:
    """Segment an image into individual digit crops.

    Returns a list of dictionaries with keys ``image``, ``bbox`` and ``area``.
    Each ``image`` is the tight digit component recentered on a square canvas,
    matching the framing used during training. Segments are sorted
    left-to-right, the reading order for form fields such as postal codes.
    """
    gray = to_grayscale(image)
    binary = binarize(gray)
    foreground = invert_if_needed(binary)

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (close_kernel_size, close_kernel_size)
    )
    cleaned = cv2.morphologyEx(foreground, cv2.MORPH_CLOSE, kernel, iterations=1)

    count, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
    segments: list[dict[str, object]] = []

    for component_id in range(1, count):
        x, y, width, height, area = stats[component_id]
        if area < min_area or width < 4 or height < 6:
            continue

        component = (labels[y : y + height, x : x + width] == component_id).astype(
            np.uint8
        ) * 255

        # A very wide component is probably two digits touching. Try to split
        # it using empty columns in its vertical projection profile.
        if width > height * 1.6:
            parts = _split_on_zero_columns(component)
            if len(parts) > 1:
                offset = x
                for part in parts:
                    part_width = int(part.shape[1])
                    segments.append(
                        {
                            "image": _center_on_canvas(part, canvas_size),
                            "bbox": (offset, int(y), part_width, int(height)),
                            "area": int((part > 0).sum()),
                        }
                    )
                    offset += part_width
                continue

        segments.append(
            {
                "image": _center_on_canvas(component, canvas_size),
                "bbox": (int(x), int(y), int(width), int(height)),
                "area": int(area),
            }
        )

    segments.sort(key=lambda item: int(item["bbox"][0]))
    return segments[:max_digits]


def draw_segmentation_boxes(
    image: np.ndarray,
    segments: list[dict[str, object]],
    color: tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    """Draw bounding boxes around detected digit components."""
    output = cv2.cvtColor(to_grayscale(image), cv2.COLOR_GRAY2BGR)
    for segment in segments:
        x, y, width, height = segment["bbox"]
        cv2.rectangle(
            output,
            (int(x), int(y)),
            (int(x + width), int(y + height)),
            color,
            thickness,
        )
    return output
